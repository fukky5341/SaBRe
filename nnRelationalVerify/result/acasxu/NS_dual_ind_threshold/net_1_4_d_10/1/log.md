## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 886.64361740241


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742)
1: (-437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521)
2: (-439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498)
3: (-536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455)
4: (-473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.62 + 2.12 = 2.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -886.6879518, upper bound: 886.6879518

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6766275, upper bound: 886.6755924
time: 0.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6733865, upper bound: 886.6733865
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -886.6766275, upper bound: 886.6755924
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -886.6733865, upper bound: 886.6733865

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -373.4723206, 606.2720947, -379.9130249, 616.8412476, -990.3135376, 986.1851196
1: -418.3924866, 599.0228271, -425.6067200, 609.5483398, -1027.9407959, 1024.6295166
2: -419.9433899, 592.4771118, -427.1958313, 602.9852905, -1022.9284668, 1019.6729126
3: -513.4045410, 690.0853882, -522.2863770, 702.1071167, -1215.5117188, 1212.3718262
4: -453.0448608, 677.0018921, -460.6833191, 688.8692017, -1141.9138184, 1137.6851807

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.03 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6733865, upper bound: 886.6733865
time: 0.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6733865, upper bound: 886.6733865
time: 0.91 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -506.5198975, 819.3720703, -366.0521545, 593.4185791, -1099.9381104, 1185.4240723
1: -573.1215820, 818.7601929, -409.9702148, 587.5056152, -1160.6271973, 1228.7303467
2: -569.8142090, 807.9828491, -411.4652100, 581.4116211, -1151.2258301, 1219.4479980
3: -704.5859375, 940.3079834, -503.2707214, 677.2180176, -1381.8038330, 1443.5784912
4: -613.9079590, 924.3903198, -442.9054565, 663.6165771, -1277.5245361, 1367.2957764

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646395, upper bound: 886.6690192
time: 0.75 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646395, upper bound: 886.6646395
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -886.6733865, upper bound: 886.6733865
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -886.6733865, upper bound: 886.6733865
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -886.6646395, upper bound: 886.6690192
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -886.6646395, upper bound: 886.6646395

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -373.4723206, 606.2720947, -373.4723206, 606.2720947, -979.7443848, 979.7443848
1: -418.3924866, 599.0228271, -418.3924866, 599.0228271, -1017.4151001, 1017.4151611
2: -419.9433899, 592.4771118, -419.9433899, 592.4771118, -1012.4205322, 1012.4204712
3: -513.4045410, 690.0853882, -513.4045410, 690.0853882, -1203.4899902, 1203.4899902
4: -453.0448608, 677.0018921, -453.0448608, 677.0018921, -1130.0467529, 1130.0467529

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.03 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6704262, upper bound: 886.6672682
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6704262, upper bound: 886.6672606
time: 0.82 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -373.4723206, 606.2720947, -506.5198975, 819.3720703, -1192.8441162, 1112.7918701
1: -418.3924866, 599.0228271, -573.1215820, 818.7601929, -1237.1527100, 1172.1444092
2: -419.9433899, 592.4771118, -569.8142090, 807.9828491, -1227.9262695, 1162.2911377
3: -513.4045410, 690.0853882, -704.5859375, 940.3079834, -1453.7124023, 1394.6711426
4: -453.0448608, 677.0018921, -613.9079590, 924.3903198, -1377.4351807, 1290.9099121

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6704262, upper bound: 886.6672682
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6711032, upper bound: 886.6672606
time: 0.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -487.4788818, 788.4369507, -339.3167114, 552.0295410, -1039.5084229, 1127.7535400
1: -551.8067017, 787.3837891, -380.5454712, 545.5744019, -1097.3811035, 1167.9291992
2: -548.5167236, 776.9584961, -382.0707703, 539.5393066, -1088.0560303, 1159.0292969
3: -678.2033081, 904.4136353, -466.5682678, 628.6406860, -1306.8438721, 1370.9819336
4: -591.2634277, 889.0295410, -412.4321594, 615.3760376, -1206.6394043, 1301.4616699

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591938, upper bound: 886.6656081
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 0.76 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -498.7020264, 807.1242676, -356.7834778, 578.4447021, -1077.1467285, 1163.9077148
1: -564.3216553, 806.2743530, -399.4561768, 572.4032593, -1136.7248535, 1205.7304688
2: -561.1560059, 795.6413574, -401.0801086, 566.5607300, -1127.7166748, 1196.7214355
3: -693.7232666, 926.1068726, -490.4295044, 659.8828735, -1353.6060791, 1416.5363770
4: -604.7384033, 910.1483765, -431.8074341, 646.5673828, -1251.3057861, 1341.9555664

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591935, upper bound: 886.6638924
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.50 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6704262, upper bound: 886.6672682
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6704262, upper bound: 886.6672606
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6704262, upper bound: 886.6672682
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6711032, upper bound: 886.6672606
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6591938, upper bound: 886.6656081
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6591935, upper bound: 886.6638924
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -349.6350098, 569.3693848, -351.7511292, 571.5529175, -921.1879272, 921.1204834
1: -392.1717224, 561.8006592, -394.1947021, 564.1727295, -956.3442993, 955.9953003
2: -393.7890320, 555.6325073, -395.7074890, 557.8972778, -951.6862793, 951.3399658
3: -480.6766052, 646.9263306, -483.3739624, 649.9805298, -1130.6567383, 1130.3000488
4: -425.8119507, 634.2166138, -427.6441345, 637.0976562, -1062.9096680, 1061.8607178

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.03 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6663788, upper bound: 886.6613050
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6824145, upper bound: 886.6808703
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -364.6583252, 591.8895874, -367.7248230, 596.8657837, -961.5241089, 959.6143799
1: -408.4007263, 584.5307007, -411.8805847, 589.5392456, -997.9399414, 996.4112549
2: -410.0544434, 578.2492065, -413.4960938, 583.1534424, -993.2077637, 991.7452393
3: -501.2101135, 673.4092407, -505.4385681, 679.1712646, -1180.3813477, 1178.8477783
4: -442.3909302, 660.6585083, -446.0850525, 666.2826538, -1108.6734619, 1106.7434082

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6734959, upper bound: 886.6662683
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816668, upper bound: 886.6816668
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -349.6350098, 569.3693848, -487.4788818, 788.4369507, -1138.0718994, 1056.8481445
1: -392.1717224, 561.8006592, -551.8067017, 787.3837891, -1179.5554199, 1113.6074219
2: -393.7890320, 555.6325073, -548.5167236, 776.9584961, -1170.7475586, 1104.1490479
3: -480.6766052, 646.9263306, -678.2033081, 904.4136353, -1385.0900879, 1325.1293945
4: -425.8119507, 634.2166138, -591.2634277, 889.0295410, -1314.8415527, 1225.4799805

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6658618, upper bound: 886.6602240
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6701795, upper bound: 886.6672682
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -364.6583252, 591.8895874, -498.7020264, 807.1242676, -1171.7825928, 1090.5915527
1: -408.4007263, 584.5307007, -564.3216553, 806.2743530, -1214.6750488, 1148.8522949
2: -410.0544434, 578.2492065, -561.1560059, 795.6413574, -1205.6956787, 1139.4052734
3: -501.2101135, 673.4092407, -693.7232666, 926.1068726, -1427.3170166, 1367.1325684
4: -442.3909302, 660.6585083, -604.7384033, 910.1483765, -1352.5390625, 1265.3969727

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693706, upper bound: 886.6620078
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6707478, upper bound: 886.6672606
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -480.5060120, 777.4941406, -326.5409851, 532.3201294, -1012.8261108, 1104.0350342
1: -544.0156250, 778.4857178, -366.1400146, 525.8981934, -1069.9135742, 1144.6256104
2: -540.3151855, 767.7396851, -367.5806885, 520.0579224, -1060.3729248, 1135.3201904
3: -669.8366699, 893.7411499, -449.1602173, 606.1613159, -1275.9979248, 1342.9012451
4: -581.6586914, 878.1356201, -396.2624207, 592.7475586, -1174.4062500, 1274.3980713

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591938, upper bound: 886.6656081
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591938, upper bound: 886.6656081
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -485.4142761, 785.1459351, -337.8146667, 549.5019531, -1034.9162598, 1122.9603271
1: -549.4979248, 784.0922241, -378.8580627, 543.1777954, -1092.6757812, 1162.9503174
2: -546.1857300, 773.6908569, -380.3740234, 537.1277466, -1083.3134766, 1154.0648193
3: -675.3648071, 900.6560669, -464.4977722, 625.8779907, -1301.2427979, 1365.1538086
4: -588.7877808, 885.2640991, -410.6450195, 612.6129761, -1201.4005127, 1295.9090576

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -490.3278809, 793.8120728, -343.1933289, 557.3472900, -1047.6750488, 1137.0052490
1: -554.9702759, 794.8746338, -384.1610413, 551.4938354, -1106.4641113, 1179.0356445
2: -551.2805176, 783.9827271, -385.6760864, 545.7434082, -1097.0239258, 1169.6585693
3: -683.3733521, 912.5440674, -472.0565186, 635.8179932, -1319.1914062, 1384.6002197
4: -593.3103638, 896.5396118, -414.7239380, 622.3978271, -1215.7081299, 1311.2634277

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591935, upper bound: 886.6638924
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591935, upper bound: 886.6638924
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -496.5157471, 803.6188965, -355.7204285, 576.6676025, -1073.1833496, 1159.3393555
1: -561.8742065, 802.7595215, -398.2504272, 570.6860352, -1132.5603027, 1201.0097656
2: -558.6856689, 792.1546631, -399.8722229, 564.8418579, -1123.5275879, 1192.0267334
3: -690.7162476, 922.0928955, -488.9558105, 657.9295044, -1348.6456299, 1411.0487061
4: -602.1058960, 906.1431274, -430.5286560, 644.5954590, -1246.7014160, 1336.6717529

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.35 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6663788, upper bound: 886.6613050
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6824145, upper bound: 886.6808703
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6734959, upper bound: 886.6662683
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6816668, upper bound: 886.6816668
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6658618, upper bound: 886.6602240
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6701795, upper bound: 886.6672682
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6693706, upper bound: 886.6620078
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6707478, upper bound: 886.6672606
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6591938, upper bound: 886.6656081
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6591938, upper bound: 886.6656081
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6591935, upper bound: 886.6638924
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6591935, upper bound: 886.6638924
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -335.4844971, 546.8212280, -325.8761597, 532.2341919, -867.7186890, 872.6973877
1: -376.2764893, 539.8469238, -366.1013489, 526.3720093, -902.6484985, 905.9482422
2: -377.7658386, 533.6960449, -366.7507935, 519.9417114, -897.7075195, 900.4468384
3: -461.4688721, 621.6394653, -449.4055786, 606.4185181, -1067.8874512, 1071.0450439
4: -408.0794983, 608.9240723, -396.5267639, 592.5942993, -1000.6738281, 1005.4508057

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6573391, upper bound: 886.6526403
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6588102, upper bound: 886.6545260
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -348.6062317, 567.7132568, -350.0037231, 568.7619629, -917.3681641, 917.7169189
1: -391.0159912, 560.1611328, -392.2479553, 561.4333496, -952.4493408, 952.4090576
2: -392.6283875, 554.0045166, -393.7348938, 555.1669312, -947.7952881, 947.7393799
3: -479.2698975, 645.0578613, -481.0073242, 646.8435059, -1126.1134033, 1126.0649414
4: -424.5896912, 632.3359375, -425.5722656, 633.9497070, -1058.5394287, 1057.9079590

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6726017, upper bound: 886.6708621
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6704152, upper bound: 886.6694836
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -350.4472046, 569.0327148, -339.5853577, 553.8225098, -904.2696533, 908.6179199
1: -392.4012451, 562.2722168, -381.1962585, 548.0220947, -940.4233398, 943.4683228
2: -393.9475403, 556.2791748, -381.9154663, 541.4406128, -935.3880615, 938.1946411
3: -481.8953552, 647.9179688, -468.2320557, 631.3611450, -1113.2563477, 1116.1500244
4: -424.5590820, 635.1586914, -412.1808777, 617.5095825, -1042.0686035, 1047.3394775

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6708626, upper bound: 886.6658557
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6675872, upper bound: 886.6631030
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -363.7988892, 590.5076294, -365.9316406, 593.9845581, -957.7834473, 956.4392090
1: -407.4398499, 583.1757202, -409.8731689, 586.7125244, -994.1523438, 993.0487671
2: -409.0846863, 576.8970337, -411.4740906, 580.3331299, -989.4178467, 988.3710938
3: -500.0372620, 671.8583984, -502.9905396, 675.9378052, -1175.9747314, 1174.8486328
4: -441.3639221, 659.1005249, -443.9406433, 663.0355835, -1104.3995361, 1103.0411377

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740773, upper bound: 886.6714973
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6695250, upper bound: 886.6695250
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -335.4844971, 546.8212280, -480.5060120, 777.4941406, -1112.9785156, 1027.3271484
1: -376.2764893, 539.8469238, -544.0156250, 778.4857178, -1154.7622070, 1083.8625488
2: -377.7658386, 533.6960449, -540.3151855, 767.7396851, -1145.5054932, 1074.0112305
3: -461.4688721, 621.6394653, -669.8366699, 893.7411499, -1355.2098389, 1291.4760742
4: -408.0794983, 608.9240723, -581.6586914, 878.1356201, -1286.2149658, 1190.5827637

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6573391, upper bound: 886.6524522
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6583009, upper bound: 886.6533244
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -348.6062317, 567.7132568, -485.4142761, 785.1459351, -1133.7520752, 1053.1275635
1: -391.0159912, 560.1611328, -549.4979248, 784.0922241, -1175.1081543, 1109.6590576
2: -392.6283875, 554.0045166, -546.1857300, 773.6908569, -1166.3192139, 1100.1901855
3: -479.2698975, 645.0578613, -675.3648071, 900.6560669, -1379.9259033, 1320.4226074
4: -424.5896912, 632.3359375, -588.7877808, 885.2640991, -1309.8537598, 1221.1237793

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6617748, upper bound: 886.6595124
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6618518, upper bound: 886.6592506
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -350.4472046, 569.0327148, -490.3278809, 793.8120728, -1144.2592773, 1059.3605957
1: -392.4012451, 562.2722168, -554.9702759, 794.8746338, -1187.2758789, 1117.2424316
2: -393.9475403, 556.2791748, -551.2805176, 783.9827271, -1177.9302979, 1107.5595703
3: -481.8953552, 647.9179688, -683.3733521, 912.5440674, -1394.4392090, 1331.2912598
4: -424.5590820, 635.1586914, -593.3103638, 896.5396118, -1321.0986328, 1228.4689941

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6605981, upper bound: 886.6600417
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6605981, upper bound: 886.6620079
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -363.7988892, 590.5076294, -496.5157471, 803.6188965, -1167.4177246, 1087.0234375
1: -407.4398499, 583.1757202, -561.8742065, 802.7595215, -1210.1993408, 1145.0499268
2: -409.0846863, 576.8970337, -558.6856689, 792.1546631, -1201.2393799, 1135.5827637
3: -500.0372620, 671.8583984, -690.7162476, 922.0928955, -1422.1298828, 1362.5744629
4: -441.3639221, 659.1005249, -602.1058960, 906.1431274, -1347.5069580, 1261.2064209

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6617235, upper bound: 886.6594743
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6627776, upper bound: 886.6594418
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -480.5060120, 777.4941406, -335.4808960, 546.8157349, -1027.3217773, 1112.9748535
1: -544.0156250, 778.4857178, -376.2724304, 539.8414307, -1083.8568115, 1154.7581787
2: -540.3151855, 767.7396851, -377.7616577, 533.6906128, -1074.0058594, 1145.5013428
3: -669.8366699, 893.7411499, -461.4640808, 621.6332397, -1291.4698486, 1355.2052002
4: -581.6586914, 878.1356201, -408.0751038, 608.9176636, -1190.5762939, 1286.2106934

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6624608
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6656081
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -480.5060120, 777.4941406, -467.7258606, 756.8996582, -1237.4056396, 1245.2198486
1: -544.0156250, 778.4857178, -529.8645630, 756.2604980, -1299.0928955, 1306.4952393
2: -540.3151855, 767.7396851, -526.4329834, 745.9671021, -1285.6872559, 1292.6639404
3: -669.8366699, 893.7411499, -651.2514038, 868.5687866, -1537.5325928, 1543.4259033
4: -581.6586914, 878.1356201, -567.7965698, 853.3966675, -1432.8851318, 1443.4069824

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6624608
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6656081
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -485.4142761, 785.1459351, -348.6026611, 567.7080078, -1053.1220703, 1133.7485352
1: -549.4979248, 784.0922241, -391.0120544, 560.1557007, -1109.6535645, 1175.1042480
2: -546.1857300, 773.6908569, -392.6243286, 553.9993286, -1100.1850586, 1166.3151855
3: -675.3648071, 900.6560669, -479.2651978, 645.0516357, -1320.4162598, 1379.9211426
4: -588.7877808, 885.2640991, -424.5853577, 632.3297729, -1221.1175537, 1309.8493652

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -485.4142761, 785.1459351, -481.5104370, 778.7850952, -1264.1993408, 1266.6562500
1: -549.4979248, 784.0922241, -545.2677002, 777.5870361, -1325.8973389, 1328.2983398
2: -546.1857300, 773.6908569, -541.9689941, 767.2659912, -1312.8013916, 1314.9505615
3: -675.3648071, 900.6560669, -669.8826904, 893.1606445, -1567.5959473, 1569.7415771
4: -588.7877808, 885.2640991, -584.6838379, 877.9452515, -1464.8894043, 1468.0770264

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -490.3278809, 793.8120728, -350.4470520, 569.0324707, -1059.3603516, 1144.2591553
1: -554.9702759, 794.8746338, -392.4010620, 562.2719116, -1117.2421875, 1187.2756348
2: -551.2805176, 783.9827271, -393.9472961, 556.2789307, -1107.5593262, 1177.9300537
3: -683.3733521, 912.5440674, -481.8950806, 647.9175415, -1331.2908936, 1394.4388428
4: -593.3103638, 896.5396118, -424.5587769, 635.1582031, -1228.4685059, 1321.0983887

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6325889, upper bound: 886.6485917
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6446901, upper bound: 886.6536694
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -490.3278809, 793.8120728, -477.7048645, 773.8536987, -1264.1816406, 1271.5166016
1: -554.9702759, 794.8746338, -540.8876953, 773.4511719, -1327.5677490, 1334.0545654
2: -551.2805176, 783.9827271, -537.5739136, 762.9464722, -1313.7845459, 1320.2158203
3: -683.3733521, 912.5440674, -665.1642456, 888.4129639, -1571.5214844, 1576.5778809
4: -593.3103638, 896.5396118, -579.3289795, 872.5759888, -1463.6107178, 1473.1778564

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6325889, upper bound: 886.6485917
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6446901, upper bound: 886.6536694
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -496.5157471, 803.6188965, -363.7986755, 590.5073242, -1087.0230713, 1167.4176025
1: -561.8742065, 802.7595215, -407.4396667, 583.1754150, -1145.0495605, 1210.1992188
2: -558.6856689, 792.1546631, -409.0844727, 576.8966675, -1135.5822754, 1201.2391357
3: -690.7162476, 922.0928955, -500.0369568, 671.8580933, -1362.5740967, 1422.1296387
4: -602.1058960, 906.1431274, -441.3636780, 659.1002197, -1261.2060547, 1347.5065918

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -496.5157471, 803.6188965, -491.1607971, 795.2998657, -1291.8155518, 1294.7796631
1: -561.8742065, 802.7595215, -555.8383179, 794.2806396, -1355.2250977, 1357.6090088
2: -558.6856689, 792.1546631, -552.7631836, 783.7797852, -1341.8591309, 1344.2706299
3: -690.7162476, 922.0928955, -683.2672729, 912.4567871, -1602.7631836, 1604.9125977
4: -602.1058960, 906.1431274, -595.8532715, 896.4211426, -1496.4096680, 1499.8543701

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.77 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6573391, upper bound: 886.6526403
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6588102, upper bound: 886.6545260
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6726017, upper bound: 886.6708621
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6704152, upper bound: 886.6694836
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6708626, upper bound: 886.6658557
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6675872, upper bound: 886.6631030
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6740773, upper bound: 886.6714973
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6695250, upper bound: 886.6695250
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6573391, upper bound: 886.6524522
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6583009, upper bound: 886.6533244
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6617748, upper bound: 886.6595124
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6618518, upper bound: 886.6592506
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6605981, upper bound: 886.6600417
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6605981, upper bound: 886.6620079
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6617235, upper bound: 886.6594743
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6627776, upper bound: 886.6594418
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6624608
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6656081
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6624608
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6513379, upper bound: 886.6656081
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6687380
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6325889, upper bound: 886.6485917
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6446901, upper bound: 886.6536694
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6325889, upper bound: 886.6485917
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6446901, upper bound: 886.6536694
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -886.6644626, upper bound: 886.6644626

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -320.9764099, 523.3057251, -318.6399841, 520.8426514, -841.8190918, 841.9456787
1: -359.8989868, 516.2771606, -358.0350037, 514.9367065, -874.8356323, 874.3121338
2: -361.4674072, 510.4380798, -358.6626282, 508.6080627, -870.0754395, 869.1005859
3: -441.4048157, 594.6195068, -439.4739990, 593.3233643, -1034.7281494, 1034.0933838
4: -390.7453308, 582.0140381, -388.0599670, 579.5361938, -970.2813721, 970.0739746

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6502138, upper bound: 886.6436609
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505477, upper bound: 886.6441368
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -328.5136108, 535.4960938, -320.8148193, 524.0419312, -852.5555420, 856.3109131
1: -368.3612061, 528.3949585, -360.3802185, 518.1186523, -886.4798584, 888.7751465
2: -369.9185181, 522.5385742, -361.0341492, 511.7963867, -881.7149048, 883.5726318
3: -451.7123718, 608.5552979, -442.4055481, 596.9595947, -1048.6718750, 1050.9606934
4: -399.7206726, 595.9675903, -390.4897156, 583.1926270, -982.9133301, 986.4572754

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6518165, upper bound: 886.6459628
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6515145, upper bound: 886.6452709
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -340.6239929, 555.3527832, -345.2073975, 561.3635864, -901.9874878, 900.5600586
1: -382.2017822, 547.4945679, -386.9905090, 553.8614502, -936.0632324, 934.4851074
2: -383.6269531, 541.5087891, -388.4168701, 547.6905518, -931.3174438, 929.9255981
3: -468.4647522, 630.6612549, -474.5578308, 638.2208862, -1106.6855469, 1105.2191162
4: -415.1404114, 617.6896362, -419.8924866, 625.2186890, -1040.3590088, 1037.5821533

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6726017, upper bound: 886.6706397
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6726017, upper bound: 886.6708621
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -357.3717041, 586.4458008, -333.2830200, 542.7637329, -900.1354370, 919.7288208
1: -400.0299988, 576.5177002, -372.6628418, 534.4452515, -934.4750977, 949.1804810
2: -402.3308716, 570.4237671, -374.7015381, 528.5543213, -930.8851929, 945.1253052
3: -490.3912964, 664.4102783, -457.0310059, 616.2832642, -1106.6745605, 1121.4410400
4: -433.0330505, 649.4640503, -403.7250061, 602.9067993, -1035.9398193, 1053.1888428

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6704152, upper bound: 886.6692901
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6704152, upper bound: 886.6694836
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -343.2666321, 557.9465332, -335.3914490, 547.3399048, -890.6065674, 893.3380127
1: -384.4906921, 550.9244385, -376.6169739, 541.4003906, -925.8911133, 927.5413818
2: -385.7667236, 544.7619629, -377.3417969, 534.8823853, -920.6489868, 922.1036377
3: -472.1138611, 634.8948975, -462.5979309, 623.8375244, -1095.9511719, 1097.4926758
4: -415.9588013, 621.9352417, -407.1903687, 609.8325195, -1025.7912598, 1029.1256104

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6674060, upper bound: 886.6594427
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6708626, upper bound: 886.6653452
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -356.7972412, 584.1713257, -321.7892456, 526.1086426, -882.9058838, 905.9605713
1: -398.7690125, 574.7973633, -360.3880920, 519.1994629, -917.9684448, 935.1853638
2: -400.9607849, 568.5834351, -361.6293640, 512.9782104, -913.9389648, 930.2127686
3: -489.5033264, 662.8660889, -442.6792297, 598.7089844, -1088.2122803, 1105.5452881
4: -430.1087646, 647.8428345, -389.0761108, 584.4054565, -1014.5140991, 1036.9189453

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6649125, upper bound: 886.6580942
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6675872, upper bound: 886.6631030
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -356.6242371, 579.4417725, -361.4518433, 587.0911865, -943.7153320, 940.8934937
1: -399.5517273, 571.8530273, -404.9489746, 579.6564331, -979.2081299, 976.8019409
2: -401.0491943, 565.6669312, -406.3994141, 573.3312378, -974.3804321, 972.0663452
3: -490.2959290, 658.9928589, -496.9120483, 667.9218140, -1158.2177734, 1155.9047852
4: -432.8025513, 645.9838257, -438.5813293, 654.8574829, -1087.6600342, 1084.5651855

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6736457, upper bound: 886.6707105
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6736457, upper bound: 886.6714910
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -370.6875916, 606.4570923, -349.0354919, 567.5398560, -938.2274170, 955.4924927
1: -414.4184265, 596.5979614, -390.0742798, 559.2715454, -973.6899414, 986.6722412
2: -416.6751709, 590.2839355, -392.2421875, 553.2752686, -969.9504395, 982.5260620
3: -508.4226074, 687.8840942, -478.7059937, 644.8922119, -1153.3146973, 1166.5900879
4: -447.4282227, 672.6928711, -421.7666626, 631.5390625, -1078.9672852, 1094.4594727

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694836, upper bound: 886.6692974
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694836, upper bound: 886.6695111
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -320.9764099, 523.3057251, -474.6433411, 768.1013184, -1089.0777588, 997.9490356
1: -359.8989868, 516.2771606, -537.3905640, 769.0122681, -1128.9112549, 1053.6676025
2: -361.4674072, 510.4380798, -533.7208862, 758.3855591, -1119.8530273, 1044.1589355
3: -441.4048157, 594.6195068, -661.7294922, 882.9371338, -1324.3419189, 1256.3488770
4: -390.7453308, 582.0140381, -574.6126099, 867.3443604, -1257.9299316, 1156.6265869

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6543020, upper bound: 886.6444265
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6543020, upper bound: 886.6524522
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -328.5136108, 535.4960938, -473.8840637, 766.8555908, -1095.3688965, 1009.3801270
1: -368.3612061, 528.3949585, -536.5441895, 767.8059082, -1136.1671143, 1064.9392090
2: -369.9185181, 522.5385742, -532.8796997, 757.1815186, -1127.0999756, 1055.4182129
3: -451.7123718, 608.5552979, -660.6470947, 881.5707397, -1333.2829590, 1269.2023926
4: -399.7206726, 595.9675903, -573.8385620, 865.8827515, -1265.6033936, 1169.8061523

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6451358
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6533244
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -333.7850647, 543.7745972, -479.6030884, 775.8255615, -1109.6105957, 1023.3776855
1: -374.2675781, 536.1069336, -542.9398804, 774.6963501, -1148.9638672, 1079.0468750
2: -376.0095520, 530.2505493, -539.6480103, 764.4119873, -1140.4215088, 1069.8985596
3: -458.7491150, 617.5330200, -667.3414307, 889.9259644, -1348.6749268, 1284.8745117
4: -406.9678955, 604.8599243, -581.8319092, 874.5728760, -1281.5404053, 1186.6918945

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6617748, upper bound: 886.6595124
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6617748, upper bound: 886.6595124
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -341.3889771, 555.9819336, -478.8054504, 774.5390015, -1115.9277344, 1034.7872314
1: -382.8246155, 548.3970337, -542.0338745, 773.4277344, -1156.2523193, 1090.4309082
2: -384.5541992, 542.3889771, -538.7780151, 763.1509399, -1147.7050781, 1081.1669922
3: -469.2173767, 631.5306396, -666.2013550, 888.5168457, -1357.7341309, 1297.7319336
4: -415.9815979, 618.9337769, -581.0110474, 873.0468750, -1289.0281982, 1199.9444580

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6618518, upper bound: 886.6592506
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6592506
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -335.8760071, 547.7655029, -490.3278809, 793.8120728, -1129.6877441, 1038.0931396
1: -376.9928589, 541.9160767, -554.9702759, 794.8746338, -1171.8674316, 1096.8863525
2: -377.7377014, 535.4390869, -551.2805176, 783.9827271, -1161.7202148, 1086.7193604
3: -463.1000977, 624.3510132, -683.3733521, 912.5440674, -1375.6436768, 1307.7243652
4: -407.6806335, 610.6179199, -593.3103638, 896.5396118, -1304.2202148, 1203.9282227

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6611451, upper bound: 886.6547863
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6559390, upper bound: 886.6560351
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6561349, upper bound: 886.6562386
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -362.8229065, 588.9429932, -490.3278809, 793.8120728, -1156.6350098, 1079.2708740
1: -406.3486328, 581.6405640, -554.9702759, 794.8746338, -1201.2232666, 1136.6108398
2: -407.9841614, 575.3658447, -551.2805176, 783.9827271, -1191.9665527, 1126.6462402
3: -498.7056580, 670.1018677, -683.3733521, 912.5440674, -1411.2495117, 1353.4749756
4: -440.1969910, 657.3359375, -593.3103638, 896.5396118, -1336.7365723, 1250.6462402

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6611451, upper bound: 886.6559326
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6491990, upper bound: 886.6501756
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6494175, upper bound: 886.6528374
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -353.8612061, 574.6835327, -490.5031433, 793.9536133, -1147.8146973, 1065.1866455
1: -396.1875305, 567.2429810, -555.0984497, 793.0181274, -1189.2056885, 1122.3414307
2: -397.9522400, 561.2069092, -551.9146118, 782.5390015, -1180.4912109, 1113.1215820
3: -486.1527710, 653.6115723, -682.4187012, 910.9588013, -1397.1113281, 1336.0302734
4: -429.6215820, 640.9849854, -594.8951416, 895.0700684, -1324.6916504, 1235.8800049

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6617235, upper bound: 886.6594743
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6617235, upper bound: 886.6594743
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -356.0942688, 578.1146240, -490.0725098, 793.3090210, -1149.4033203, 1068.1871338
1: -398.7055054, 570.7373047, -554.5926514, 792.4016724, -1191.1071777, 1125.3299561
2: -400.4404297, 564.6192017, -551.4684448, 781.9148560, -1182.3552246, 1116.0872803
3: -489.3798523, 657.6080933, -681.7733765, 910.3055420, -1399.6853027, 1339.3813477
4: -432.2027588, 644.9402466, -594.5486450, 894.2377319, -1326.4404297, 1239.4888916

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6625907, upper bound: 886.6594418
time: 2.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6625907, upper bound: 886.6594418
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -476.7803650, 771.6308594, -335.4808960, 546.8157349, -1023.5960083, 1107.1114502
1: -539.9763184, 772.4961548, -376.2724304, 539.8414307, -1079.8176270, 1148.7685547
2: -536.3087158, 761.8306885, -377.7616577, 533.6906128, -1069.9992676, 1139.5922852
3: -664.6658936, 886.8717041, -461.4640808, 621.6332397, -1286.2990723, 1348.3358154
4: -577.7337036, 871.2992554, -408.0751038, 608.9176636, -1186.6511230, 1279.3741455

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6523146, upper bound: 886.6626855
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6526090, upper bound: 886.6624196
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -483.8485718, 783.6398926, -335.4808960, 546.8157349, -1030.6643066, 1119.1207275
1: -547.6936646, 784.4992065, -376.2724304, 539.8414307, -1087.5350342, 1160.7716064
2: -544.0818481, 773.7381592, -377.7616577, 533.6906128, -1077.7724609, 1151.4997559
3: -674.3672485, 900.7269897, -461.4640808, 621.6332397, -1296.0004883, 1362.1910400
4: -585.6768188, 884.7250366, -408.0751038, 608.9176636, -1194.5943604, 1292.8000488

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6523146, upper bound: 886.6658618
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6526090, upper bound: 886.6624196
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -476.7803650, 771.6308594, -467.7258606, 756.8996582, -1233.6800537, 1239.3566895
1: -539.9763184, 772.4961548, -529.8645630, 756.2604980, -1295.0684814, 1300.4575195
2: -536.3087158, 761.8306885, -526.4329834, 745.9671021, -1281.6440430, 1286.7365723
3: -664.6658936, 886.8717041, -651.2514038, 868.5687866, -1532.3648682, 1536.5111084
4: -577.7337036, 871.2992554, -567.7965698, 853.3966675, -1428.8464355, 1436.4322510

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6478311
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6624608
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -483.8485718, 783.6398926, -467.7258606, 756.8996582, -1240.7482910, 1251.3657227
1: -547.6936646, 784.4992065, -529.8645630, 756.2604980, -1302.7553711, 1312.5164795
2: -544.0818481, 773.7381592, -526.4329834, 745.9671021, -1289.4361572, 1298.6812744
3: -674.3672485, 900.7269897, -651.2514038, 868.5687866, -1542.0844727, 1550.4611816
4: -585.6768188, 884.7250366, -567.7965698, 853.3966675, -1436.7575684, 1449.8769531

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6478311
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6656081
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -348.6026611, 567.7080078, -1047.9545898, 1125.3729248
1: -543.8629150, 775.5766602, -391.0120544, 560.1557007, -1104.0185547, 1166.5887451
2: -540.5451050, 765.2640991, -392.6243286, 553.9993286, -1094.5443115, 1157.8883057
3: -668.1536255, 890.8625488, -479.2651978, 645.0516357, -1313.2049561, 1370.1276855
4: -583.1776733, 875.6425781, -424.5853577, 632.3297729, -1215.5074463, 1300.2277832

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6595124, upper bound: 886.6617748
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6451358, upper bound: 886.6618518
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -348.6026611, 567.7080078, -1058.2006836, 1142.8317871
1: -555.0930786, 793.1872559, -391.0120544, 560.1557007, -1115.2487793, 1184.1993408
2: -552.0131836, 782.6972656, -392.6243286, 553.9993286, -1106.0122070, 1175.3215332
3: -682.3418579, 911.2104492, -479.2651978, 645.0516357, -1327.3933105, 1390.4755859
4: -595.0556030, 895.1871338, -424.5853577, 632.3297729, -1227.3853760, 1319.7724609

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6444265, upper bound: 886.6617748
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592506, upper bound: 886.6618518
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -481.5104370, 778.7850952, -1259.0318604, 1258.2805176
1: -543.8629150, 775.5766602, -545.2677002, 777.5870361, -1320.2843018, 1319.6169434
2: -540.5451050, 765.2640991, -541.9689941, 767.2659912, -1307.0750732, 1306.4234619
3: -668.1536255, 890.8625488, -669.8826904, 893.1606445, -1560.3583984, 1559.7504883
4: -583.1776733, 875.6425781, -584.6838379, 877.9452515, -1459.1301270, 1458.2941895

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6478311
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6644784
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -481.5104370, 778.7850952, -1269.2780762, 1275.7395020
1: -555.0930786, 793.1872559, -545.2677002, 777.5870361, -1331.4843750, 1337.3543701
2: -552.0131836, 782.6972656, -541.9689941, 767.2659912, -1318.5826416, 1323.9125977
3: -682.3418579, 911.2104492, -669.8826904, 893.1606445, -1574.6119385, 1580.2923584
4: -595.0556030, 895.1871338, -584.6838379, 877.9452515, -1470.8911133, 1477.8173828

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6478311
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6644784
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -455.0776672, 735.9932861, -323.3966064, 525.6923218, -980.7700195, 1059.3896484
1: -514.1742554, 737.5162354, -361.5787964, 519.8642578, -1034.0385742, 1099.0949707
2: -510.8902893, 727.1378784, -363.3972168, 514.3818970, -1025.2722168, 1090.5351562
3: -635.0394897, 848.2443237, -444.7953491, 599.4934692, -1234.5329590, 1293.0396729
4: -548.4417725, 831.5809326, -390.8852234, 586.7935181, -1135.2353516, 1222.2054443

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6314809, upper bound: 886.6419427
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6314809, upper bound: 886.6528675
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -488.3928223, 790.5830078, -348.2615051, 565.5121460, -1053.9050293, 1138.8444824
1: -552.8066406, 791.6104736, -389.9071350, 558.6245728, -1111.4310303, 1181.5175781
2: -549.0991211, 780.7411499, -391.4711304, 552.4727173, -1101.5716553, 1172.2122803
3: -680.7133179, 908.8287354, -478.8070679, 643.6444092, -1324.3576660, 1387.6356201
4: -590.9399414, 892.8865967, -421.8570557, 630.9657593, -1221.9057617, 1314.7435303

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6433475, upper bound: 886.6440667
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6433481, upper bound: 886.6597555
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -455.0776672, 735.9932861, -443.8610840, 719.9006348, -1174.9782715, 1179.8540039
1: -514.1742554, 737.5162354, -502.6467590, 719.8716431, -1233.1193848, 1238.0831299
2: -510.8902893, 727.1378784, -499.6049805, 709.8674927, -1220.0076904, 1225.1093750
3: -635.0394897, 848.2443237, -618.5871582, 827.3850708, -1461.7943115, 1465.6418457
4: -548.4417725, 831.5809326, -538.3806763, 811.6503296, -1357.3536377, 1366.9676514

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6235888, upper bound: 886.6406260
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6312320, upper bound: 886.6485917
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -488.3928223, 790.5830078, -476.8112488, 772.3640747, -1260.7568359, 1267.3942871
1: -552.8066406, 791.6104736, -539.8874512, 771.9406738, -1323.8468018, 1329.7244873
2: -549.0991211, 780.7411499, -536.5659180, 761.4447021, -1310.0032959, 1315.8811035
3: -680.7133179, 908.8287354, -663.9331055, 886.6938477, -1566.9869385, 1571.5064697
4: -590.9399414, 892.8865967, -578.2315674, 870.8782959, -1459.5220947, 1468.4172363

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6430551, upper bound: 886.6430551
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6430551, upper bound: 886.6536694
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -363.7986755, 590.5073242, -1070.7539062, 1140.5689697
1: -543.8629150, 775.5766602, -407.4396667, 583.1754150, -1127.0382080, 1183.0163574
2: -540.5451050, 765.2640991, -409.0844727, 576.8966675, -1117.4417725, 1174.3486328
3: -668.1536255, 890.8625488, -500.0369568, 671.8580933, -1340.0115967, 1390.8994141
4: -583.1776733, 875.6425781, -441.3636780, 659.1002197, -1242.2778320, 1317.0061035

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594743, upper bound: 886.6617235
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592506, upper bound: 886.6625907
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -363.7986755, 590.5073242, -1081.0002441, 1158.0277100
1: -555.0930786, 793.1872559, -407.4396667, 583.1754150, -1138.2684326, 1200.6269531
2: -552.0131836, 782.6972656, -409.0844727, 576.8966675, -1128.9097900, 1191.7817383
3: -682.3418579, 911.2104492, -500.0369568, 671.8580933, -1354.1998291, 1411.2473145
4: -595.0556030, 895.1871338, -441.3636780, 659.1002197, -1254.1557617, 1336.5507812

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594743, upper bound: 886.6617019
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592506, upper bound: 886.6623341
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -491.1607971, 795.2998657, -1275.5466309, 1267.9309082
1: -543.8629150, 775.5766602, -555.8383179, 794.2806396, -1337.1241455, 1330.1574707
2: -540.5451050, 765.2640991, -552.7631836, 783.7797852, -1323.6468506, 1317.2634277
3: -668.1536255, 890.8625488, -683.2672729, 912.4567871, -1579.8781738, 1573.1986084
4: -583.1776733, 875.6425781, -595.8532715, 896.4211426, -1477.5792236, 1469.3753662

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6591935
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6642566
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -491.1607971, 795.2998657, -1285.7928467, 1285.3898926
1: -555.0930786, 793.1872559, -555.8383179, 794.2806396, -1348.4501953, 1348.0157471
2: -552.0131836, 782.6972656, -552.7631836, 783.7797852, -1335.1745605, 1334.7738037
3: -682.3418579, 911.2104492, -683.2672729, 912.4567871, -1594.4151611, 1594.0235596
4: -595.0556030, 895.1871338, -595.8532715, 896.4211426, -1489.3402100, 1488.8985596

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6582171
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6642566
time: 1.00 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.06 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6502138, upper bound: 886.6436609
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6505477, upper bound: 886.6441368
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6518165, upper bound: 886.6459628
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6515145, upper bound: 886.6452709
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6726017, upper bound: 886.6706397
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6726017, upper bound: 886.6708621
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6704152, upper bound: 886.6692901
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6704152, upper bound: 886.6694836
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6674060, upper bound: 886.6594427
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6708626, upper bound: 886.6653452
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6649125, upper bound: 886.6580942
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6675872, upper bound: 886.6631030
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6736457, upper bound: 886.6707105
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6736457, upper bound: 886.6714910
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6694836, upper bound: 886.6692974
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6694836, upper bound: 886.6695111
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6543020, upper bound: 886.6444265
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6543020, upper bound: 886.6524522
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6451358
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6533244
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6617748, upper bound: 886.6595124
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6617748, upper bound: 886.6595124
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6618518, upper bound: 886.6592506
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6592506
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6559390, upper bound: 886.6560351
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6561349, upper bound: 886.6562386
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6491990, upper bound: 886.6501756
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6494175, upper bound: 886.6528374
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6617235, upper bound: 886.6594743
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6617235, upper bound: 886.6594743
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6625907, upper bound: 886.6594418
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6625907, upper bound: 886.6594418
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6523146, upper bound: 886.6626855
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6526090, upper bound: 886.6624196
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6523146, upper bound: 886.6658618
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6526090, upper bound: 886.6624196
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6478311
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6624608
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6478311
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6656081
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6595124, upper bound: 886.6617748
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6451358, upper bound: 886.6618518
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6444265, upper bound: 886.6617748
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6592506, upper bound: 886.6618518
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6478311, upper bound: 886.6478311
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6644784
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6478311
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6644784
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6314809, upper bound: 886.6419427
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6314809, upper bound: 886.6528675
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6433475, upper bound: 886.6440667
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6433481, upper bound: 886.6597555
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6235888, upper bound: 886.6406260
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6312320, upper bound: 886.6485917
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6430551, upper bound: 886.6430551
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6430551, upper bound: 886.6536694
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6594743, upper bound: 886.6617235
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6592506, upper bound: 886.6625907
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6594743, upper bound: 886.6617019
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6592506, upper bound: 886.6623341
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6591935
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509900, upper bound: 886.6642566
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6582171
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6642566

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -308.7111511, 503.1630859, -294.2297363, 481.3695374, -790.0806274, 797.3928223
1: -345.8934326, 495.9386292, -330.3752747, 475.0233459, -820.9167480, 826.3139038
2: -347.6737671, 490.5121765, -331.1929016, 469.4813232, -817.1550293, 821.7050781
3: -424.2344055, 571.1401367, -405.6489563, 547.3903198, -971.6245728, 976.7889404
4: -375.7129211, 559.2048340, -358.3760071, 534.8362427, -910.5490112, 917.5808105

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -311.1076660, 507.5102234, -302.8676147, 495.7476501, -806.8552856, 810.3778076
1: -348.9346924, 500.5705872, -340.5835876, 489.8650818, -838.7998047, 841.1541748
2: -350.4716187, 494.8428650, -341.1480408, 483.7907410, -834.2623291, 835.9909058
3: -427.7901611, 576.5418091, -417.7852783, 564.4152222, -992.2053833, 994.3269653
4: -379.2967529, 564.1493530, -369.7783203, 551.2199097, -930.5166016, 933.9276733

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -317.1283569, 516.6817017, -296.2059021, 484.3117371, -801.4398804, 812.8875732
1: -355.3772888, 509.4357300, -332.4775391, 477.9424438, -833.3197021, 841.9131470
2: -357.1178589, 504.0725403, -333.3239746, 472.3874512, -829.5052490, 837.3964844
3: -435.7681274, 586.8249512, -408.3214722, 550.7556152, -986.5235596, 995.1464233
4: -385.7456970, 574.8767700, -360.5585632, 538.1074829, -923.8530884, 935.4353027

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6518165, upper bound: 886.6450779
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6515394, upper bound: 886.6457548
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -318.1214600, 518.8888550, -303.8962708, 497.0166931, -815.1381226, 822.7851562
1: -356.8121338, 511.8807983, -341.6340332, 491.1337891, -847.9458618, 853.5148315
2: -358.3401489, 506.0228577, -342.2334290, 485.0921326, -843.4321289, 848.2562256
3: -437.3858643, 589.4555054, -419.1253052, 565.8661499, -1003.2520142, 1008.5808105
4: -387.6566772, 577.1032104, -370.8247070, 552.6992188, -940.3558350, 947.9278564

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6515145, upper bound: 886.6441890
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6512756, upper bound: 886.6445389
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -340.6239929, 555.3527832, -342.3125305, 557.8886719, -898.5125732, 897.6652832
1: -382.2017822, 547.4945679, -384.0537109, 550.1743774, -932.3761597, 931.5482788
2: -383.6269531, 541.5087891, -385.4885559, 544.1345215, -927.7614136, 926.9972534
3: -468.4647522, 630.6612549, -470.7468262, 633.6989136, -1102.1636963, 1101.4075928
4: -415.1404114, 617.6896362, -417.1504211, 620.8044434, -1035.9445801, 1034.8400879

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6645052, upper bound: 886.6624638
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6623632, upper bound: 886.6614060
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -340.6239929, 555.3527832, -358.3032227, 581.9827881, -922.6068115, 913.6558228
1: -382.2017822, 547.4945679, -401.3855286, 574.5154419, -956.7172241, 948.8801270
2: -383.6269531, 541.5087891, -402.8547668, 568.2897949, -951.9166870, 944.3635254
3: -468.4647522, 630.6612549, -492.5768433, 662.0063477, -1130.4709473, 1123.2380371
4: -415.1404114, 617.6896362, -434.7894592, 649.0710449, -1064.2113037, 1052.4791260

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6645052, upper bound: 886.6631277
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6623632, upper bound: 886.6622589
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -357.3717041, 586.4458008, -333.1895447, 543.7777100, -901.1494141, 919.6352539
1: -400.0299988, 576.5177002, -372.9783630, 535.3852539, -935.4151001, 949.4958496
2: -402.3308716, 570.4237671, -375.0880737, 529.5178833, -931.8487549, 945.5118408
3: -490.3912964, 664.4102783, -457.2063904, 616.9902344, -1107.3813477, 1121.6166992
4: -433.0330505, 649.4640503, -404.3413696, 603.8210449, -1036.8541260, 1053.8054199

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628161, upper bound: 886.6609012
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609566, upper bound: 886.6609108
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -357.3717041, 586.4458008, -346.0719299, 562.7155762, -920.0872803, 932.5176392
1: -400.0299988, 576.5177002, -386.7100220, 554.4066772, -954.4366455, 963.2277222
2: -402.3308716, 570.4237671, -388.9108887, 548.5159912, -950.8468628, 959.3346558
3: -490.3912964, 664.4102783, -474.6221619, 639.2850342, -1129.6762695, 1139.0319824
4: -433.0330505, 649.4640503, -418.1769409, 626.0872192, -1059.1202393, 1067.6409912

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628161, upper bound: 886.6618703
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609566, upper bound: 886.6619908
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -334.4762878, 544.8044434, -284.3964539, 468.6184692, -803.0947266, 829.2009277
1: -374.5053711, 537.4190674, -318.7785645, 461.9045105, -836.4098511, 856.1976318
2: -375.8716431, 531.4118652, -319.7507019, 456.0436096, -831.9152832, 851.1623535
3: -460.0002136, 619.4482422, -392.1552124, 533.1889038, -993.1890869, 1011.6034546
4: -405.3483276, 606.3789673, -344.9452820, 518.3757324, -923.7240601, 951.3242188

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6599447, upper bound: 886.6564801
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6599447, upper bound: 886.6594427
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -338.4449463, 549.7628174, -328.9003601, 536.1521606, -874.5971069, 878.6632080
1: -379.1168823, 542.8088379, -369.3868103, 530.3654785, -909.4823608, 912.1956787
2: -380.3739624, 536.7061768, -370.0636902, 523.9140015, -904.2878418, 906.7698975
3: -465.4830322, 625.5643311, -453.6897278, 611.1530151, -1076.6359863, 1079.2540283
4: -410.1832886, 612.8538208, -399.3677368, 597.5227661, -1007.7060547, 1012.2214966

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6611485, upper bound: 886.6617156
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6611485, upper bound: 886.6653452
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -351.6496582, 576.2152100, -274.5028381, 453.3698730, -805.0195312, 850.7180176
1: -392.9066772, 566.6397095, -306.7852783, 445.9661865, -838.8728638, 873.4249878
2: -395.1482544, 560.5625000, -308.4790649, 440.3464355, -835.4946289, 869.0415039
3: -482.3772278, 653.5479736, -377.5926208, 515.1694336, -997.5466309, 1031.1406250
4: -423.8818054, 638.4639893, -331.7532959, 500.2744751, -924.1562500, 970.2172241

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6601859, upper bound: 886.6565223
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6601859, upper bound: 886.6580942
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -349.6139832, 572.6589355, -316.1160583, 516.1474609, -865.7614746, 888.7750244
1: -390.8705750, 563.3460693, -354.0531921, 509.4178467, -900.2884521, 917.3992310
2: -392.9483032, 557.2003784, -355.2511292, 503.2583008, -896.2066040, 912.4514771
3: -479.7066650, 649.6468506, -434.8657227, 587.4653320, -1067.1719971, 1084.5125732
4: -421.7407227, 634.9502563, -382.1953735, 573.5797729, -995.3204956, 1017.1456299

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613888, upper bound: 886.6613888
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613888, upper bound: 886.6631030
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -356.6242371, 579.4417725, -342.2335205, 557.7585449, -914.3826294, 921.6752930
1: -399.5517273, 571.8530273, -383.9642029, 550.0468750, -949.5985718, 955.8172607
2: -401.0491943, 565.6669312, -385.3991394, 544.0076294, -945.0568237, 951.0660400
3: -490.2959290, 658.9928589, -470.6368713, 633.5531616, -1123.8491211, 1129.6297607
4: -432.8025513, 645.9838257, -417.0513000, 620.6575928, -1053.4600830, 1063.0351562

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6650601, upper bound: 886.6624638
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6651941, upper bound: 886.6624736
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -356.6242371, 579.4417725, -358.3032227, 581.9827881, -938.6069336, 937.7448120
1: -399.5517273, 571.8530273, -401.3855286, 574.5154419, -974.0671387, 973.2385254
2: -401.0491943, 565.6669312, -402.8547668, 568.2897949, -969.3389893, 968.5217285
3: -490.2959290, 658.9928589, -492.5768433, 662.0063477, -1152.3022461, 1151.5697021
4: -432.8025513, 645.9838257, -434.7894592, 649.0710449, -1081.8735352, 1080.7733154

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6650601, upper bound: 886.6634401
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6732964, upper bound: 886.6709525
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716239, upper bound: 886.6698204
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -370.6875916, 606.4570923, -333.1594543, 543.7278442, -914.4154053, 939.6165161
1: -414.4184265, 596.5979614, -372.9441833, 535.3364868, -949.7548828, 969.5421143
2: -416.6751709, 590.2839355, -375.0538635, 529.4693604, -946.1445312, 965.3377686
3: -508.4226074, 687.8840942, -457.1645813, 616.9345703, -1125.3569336, 1145.0487061
4: -447.4282227, 672.6928711, -404.3038330, 603.7649536, -1051.1931152, 1076.9967041

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6560815, upper bound: 886.6575616
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6619908, upper bound: 886.6609108
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -370.6875916, 606.4570923, -346.0719299, 562.7155762, -933.4031372, 952.5288696
1: -414.4184265, 596.5979614, -386.7100220, 554.4066772, -968.8250732, 983.3079834
2: -416.6751709, 590.2839355, -388.9108887, 548.5159912, -965.1911621, 979.1948242
3: -508.4226074, 687.8840942, -474.6221619, 639.2850342, -1147.7076416, 1162.5059814
4: -447.4282227, 672.6928711, -418.1769409, 626.0872192, -1073.5153809, 1090.8698730

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6560815, upper bound: 886.6575616
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6619908, upper bound: 886.6621163
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -320.9764099, 523.3057251, -470.1524353, 760.9535522, -1081.9299316, 993.4581299
1: -359.8989868, 516.2771606, -532.4943848, 761.7310181, -1121.6300049, 1048.7714844
2: -361.4674072, 510.4380798, -528.8496094, 751.1856689, -1112.6530762, 1039.2875977
3: -441.4048157, 594.6195068, -655.4832153, 874.6005859, -1316.0052490, 1250.1025391
4: -390.7453308, 582.0140381, -569.7890625, 859.0182495, -1249.4570312, 1151.8031006

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -320.9764099, 523.3057251, -477.7374268, 773.8582764, -1094.8347168, 1001.0431519
1: -359.8989868, 516.2771606, -540.7930298, 774.6431885, -1134.5421143, 1057.0701904
2: -361.4674072, 510.4380798, -537.2103271, 763.9965210, -1125.4638672, 1047.6481934
3: -441.4048157, 594.6195068, -665.9199829, 889.4844360, -1330.8891602, 1260.5391846
4: -390.7453308, 582.0140381, -578.3269043, 873.4786377, -1263.9599609, 1160.3409424

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -328.5136108, 535.4960938, -470.5243225, 761.6109009, -1090.1245117, 1006.0203857
1: -368.3612061, 528.3949585, -532.9121704, 762.4204102, -1130.7816162, 1061.3071289
2: -369.9185181, 522.5385742, -529.2894287, 751.8820190, -1121.8005371, 1051.8280029
3: -451.7123718, 608.5552979, -655.9882812, 875.3892822, -1327.1015625, 1264.5435791
4: -399.7206726, 595.9675903, -570.3353882, 859.7593384, -1259.4799805, 1166.3029785

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6448459
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6548275, upper bound: 886.6451358
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -328.5136108, 535.4960938, -477.4057312, 773.2810059, -1101.7946777, 1012.9018555
1: -368.3612061, 528.3949585, -540.4204712, 774.1023560, -1142.4636230, 1068.8154297
2: -369.9185181, 522.5385742, -536.8490601, 763.4667358, -1133.3851318, 1059.3876953
3: -451.7123718, 608.5552979, -665.4168091, 888.8782959, -1340.5905762, 1273.9720459
4: -399.7206726, 595.9675903, -578.0786743, 872.8040161, -1272.5244141, 1174.0462646

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6524587
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6548275, upper bound: 886.6528114
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -333.7850647, 543.7745972, -473.6373901, 766.1722412, -1099.9572754, 1017.4119873
1: -374.2675781, 536.1069336, -536.4125366, 764.9110718, -1139.1787109, 1072.5194092
2: -376.0095520, 530.2505493, -533.1043701, 754.7048340, -1130.7143555, 1063.3549805
3: -458.7491150, 617.5330200, -659.0275879, 878.6658936, -1337.4150391, 1276.5605469
4: -406.9678955, 604.8599243, -575.2755737, 863.4638062, -1270.4316406, 1180.1354980

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591596, upper bound: 886.6579273
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6612640, upper bound: 886.6587209
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -333.7850647, 543.7745972, -484.4251404, 784.4790039, -1118.2640381, 1028.1997070
1: -374.2675781, 536.1069336, -548.2562866, 783.3593140, -1157.6268311, 1084.3632812
2: -376.0095520, 530.2505493, -545.1821289, 772.9958496, -1149.0052490, 1075.4326172
3: -458.7491150, 617.5330200, -673.9692993, 899.9773560, -1358.7260742, 1291.5023193
4: -406.9678955, 604.8599243, -587.7809448, 884.0230103, -1290.9909668, 1192.6408691

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591596, upper bound: 886.6583471
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6612640, upper bound: 886.6587209
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -341.3889771, 555.9819336, -474.0287781, 766.7897339, -1108.1784668, 1030.0104980
1: -382.8246155, 548.3970337, -536.8328857, 765.5568848, -1148.3814697, 1085.2299805
2: -384.5541992, 542.3889771, -533.5804443, 755.3500977, -1139.9041748, 1075.9693604
3: -469.2173767, 631.5306396, -659.5257568, 879.4155884, -1348.6328125, 1291.0561523
4: -415.9815979, 618.9337769, -575.8323975, 864.1560669, -1280.1373291, 1194.7659912

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6618128, upper bound: 886.6590792
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6612420, upper bound: 886.6590786
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -341.3889771, 555.9819336, -484.0476685, 783.9137573, -1125.3024902, 1040.0295410
1: -382.8246155, 548.3970337, -547.8065186, 782.8142700, -1165.6389160, 1096.2034912
2: -384.5541992, 542.3889771, -544.7919312, 772.4473877, -1157.0015869, 1087.1809082
3: -469.2173767, 631.5306396, -673.3983154, 899.4086914, -1368.6260986, 1304.9289551
4: -415.9815979, 618.9337769, -587.4901123, 883.2806396, -1299.2619629, 1206.4238281

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6618128, upper bound: 886.6590792
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6612420, upper bound: 886.6590786
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -334.2312012, 545.1735840, -486.4317017, 787.4518433, -1121.6831055, 1031.6052246
1: -375.1237488, 539.3024902, -550.6191406, 788.5685425, -1163.6922607, 1089.9215088
2: -375.8638000, 532.8591309, -546.8823242, 777.6978760, -1153.5614014, 1079.7414551
3: -460.8292847, 621.3460083, -677.9813843, 905.3099976, -1366.1390381, 1299.3271484
4: -405.6588440, 607.6661987, -588.7299194, 889.3002930, -1294.9591064, 1196.3959961

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6432996, upper bound: 886.6457827
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6460482, upper bound: 886.6462344
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -326.9670715, 533.6384277, -483.1061096, 784.0027466, -1110.9697266, 1016.7445068
1: -366.8065491, 527.7897949, -546.2978516, 784.6925049, -1151.4989014, 1074.0876465
2: -367.6459961, 521.3802490, -543.2014771, 773.7844238, -1141.4304199, 1064.5816650
3: -450.6699829, 608.1871338, -673.1204224, 901.2124023, -1351.8819580, 1281.3076172
4: -396.8380737, 594.3319092, -584.5404663, 884.4576416, -1281.2954102, 1178.8719482

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6436763, upper bound: 886.6461614
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6561349, upper bound: 886.6562386
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6561349, upper bound: 886.6562386
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -355.1595154, 576.4279785, -471.6601562, 764.3767700, -1119.5362549, 1048.0881348
1: -397.5652771, 568.9184570, -533.9210815, 764.8789673, -1162.4442139, 1102.8392334
2: -399.3745117, 562.9859619, -530.4498291, 754.5184937, -1153.8928223, 1093.4356689
3: -487.9959717, 655.4614868, -657.4689941, 878.0736084, -1366.0694580, 1312.9304199
4: -430.7838135, 643.2786865, -571.2269897, 862.7580566, -1293.5417480, 1214.5054932

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6501756
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6501756
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -348.1813049, 565.2661743, -470.6861267, 761.8865967, -1110.0677490, 1035.9522705
1: -390.0816956, 558.0282593, -532.8744507, 762.5156860, -1152.5972900, 1090.9027100
2: -391.6778870, 551.9182739, -529.2604980, 752.0961914, -1143.7739258, 1081.1787109
3: -478.4626770, 642.8818970, -656.1401367, 875.4735107, -1353.9359131, 1299.0218506
4: -423.0051270, 630.4554443, -569.6632080, 860.2778320, -1283.2365723, 1200.1186523

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6527202
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6528374
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -353.8612061, 574.6835327, -473.6373901, 766.1722412, -1120.0333252, 1048.3209229
1: -396.1875305, 567.2429810, -536.4125366, 764.9110718, -1161.0986328, 1103.6555176
2: -397.9522400, 561.2069092, -533.1043701, 754.7048340, -1152.6568604, 1094.3112793
3: -486.1527710, 653.6115723, -659.0275879, 878.6658936, -1364.8186035, 1312.6391602
4: -429.6215820, 640.9849854, -575.2755737, 863.4638062, -1293.0854492, 1216.2604980

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6608412, upper bound: 886.6579776
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6612236, upper bound: 886.6586617
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -353.8612061, 574.6835327, -484.4251404, 784.4790039, -1138.3402100, 1059.1086426
1: -396.1875305, 567.2429810, -548.2562866, 783.3593140, -1179.5468750, 1115.4992676
2: -397.9522400, 561.2069092, -545.1821289, 772.9958496, -1170.9477539, 1106.3890381
3: -486.1527710, 653.6115723, -673.9692993, 899.9773560, -1386.1298828, 1327.5808105
4: -429.6215820, 640.9849854, -587.7809448, 884.0230103, -1313.6445312, 1228.7657471

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6608412, upper bound: 886.6583763
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6612236, upper bound: 886.6586617
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -356.0942688, 578.1146240, -474.0287781, 766.7897339, -1122.8840332, 1052.1434326
1: -398.7055054, 570.7373047, -536.8328857, 765.5568848, -1164.2619629, 1107.5701904
2: -400.4404297, 564.6192017, -533.5804443, 755.3500977, -1155.7904053, 1098.1993408
3: -489.3798523, 657.6080933, -659.5257568, 879.4155884, -1368.7952881, 1317.1335449
4: -432.2027588, 644.9402466, -575.8323975, 864.1560669, -1296.3588867, 1220.7724609

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6620156, upper bound: 886.6582093
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6619850, upper bound: 886.6586120
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -356.0942688, 578.1146240, -484.0476685, 783.9137573, -1140.0080566, 1062.1623535
1: -398.7055054, 570.7373047, -547.8065186, 782.8142700, -1181.5197754, 1118.5438232
2: -400.4404297, 564.6192017, -544.7919312, 772.4473877, -1172.8878174, 1109.4110107
3: -489.3798523, 657.6080933, -673.3983154, 899.4086914, -1388.7884521, 1331.0063477
4: -432.2027588, 644.9402466, -587.4901123, 883.2806396, -1315.4833984, 1232.4304199

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6620156, upper bound: 886.6584902
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6619850, upper bound: 886.6586120
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -476.7803650, 771.6308594, -331.1836853, 539.8295288, -1016.6097412, 1102.8144531
1: -539.9763184, 772.4961548, -371.4457703, 532.9843140, -1072.9606934, 1143.9418945
2: -536.3087158, 761.8306885, -372.9130859, 526.8737183, -1063.1823730, 1134.7435303
3: -664.6658936, 886.8717041, -455.6068115, 613.7391968, -1278.4049072, 1342.4783936
4: -577.7337036, 871.2992554, -402.8946228, 601.1984253, -1178.9320068, 1274.1934814

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -476.7803650, 771.6308594, -334.9705200, 545.9664307, -1022.7467041, 1106.6011963
1: -539.9763184, 772.4961548, -375.6965332, 538.9942017, -1078.9704590, 1148.1926270
2: -536.3087158, 761.8306885, -377.1854248, 532.8515625, -1069.1602783, 1139.0159912
3: -664.6658936, 886.8717041, -460.7552795, 620.6595459, -1285.3254395, 1347.6268311
4: -577.7337036, 871.2992554, -407.4533081, 607.9724121, -1185.7060547, 1278.7525635

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -483.8485718, 783.6398926, -331.1836853, 539.8295288, -1023.6780396, 1114.8236084
1: -547.6936646, 784.4992065, -371.4457703, 532.9843140, -1080.6779785, 1155.9449463
2: -544.0818481, 773.7381592, -372.9130859, 526.8737183, -1070.9554443, 1146.6508789
3: -674.3672485, 900.7269897, -455.6068115, 613.7391968, -1288.1060791, 1356.3337402
4: -585.6768188, 884.7250366, -402.8946228, 601.1984253, -1186.8752441, 1287.6191406

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594218, upper bound: 886.6652353
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594218, upper bound: 886.6656018
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -483.8485718, 783.6398926, -334.9705200, 545.9664307, -1029.8149414, 1118.6103516
1: -547.6936646, 784.4992065, -375.6965332, 538.9942017, -1086.6878662, 1160.1958008
2: -544.0818481, 773.7381592, -377.1854248, 532.8515625, -1076.9333496, 1150.9233398
3: -674.3672485, 900.7269897, -460.7552795, 620.6595459, -1295.0268555, 1361.4822998
4: -585.6768188, 884.7250366, -407.4533081, 607.9724121, -1193.6491699, 1292.1783447

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6597438, upper bound: 886.6652353
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6597438, upper bound: 886.6656019
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -476.7803650, 771.6308594, -475.7322998, 769.9331055, -1246.4423828, 1247.0974121
1: -539.9763184, 772.4961548, -538.7338257, 770.7523804, -1308.3994141, 1308.9117432
2: -536.3087158, 761.8306885, -535.1314087, 760.0953979, -1294.5133057, 1295.0843506
3: -664.6658936, 886.8717041, -663.1502686, 884.9095459, -1547.2509766, 1547.7121582
4: -577.7337036, 871.2992554, -576.4880981, 869.2595215, -1444.0607910, 1444.8493652

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -476.7803650, 771.6308594, -479.0361633, 774.8145142, -1251.5948486, 1250.6669922
1: -539.9763184, 772.4961548, -542.4277954, 773.5675659, -1311.9708252, 1312.8862305
2: -536.3087158, 761.8306885, -539.1839600, 763.2622681, -1298.5483398, 1299.2652588
3: -664.6658936, 886.8717041, -666.4015503, 888.5993042, -1551.8386230, 1551.2664795
4: -577.7337036, 871.2992554, -581.7400513, 873.2807007, -1448.6500244, 1450.3645020

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -483.8485718, 783.6398926, -475.7322998, 769.9331055, -1253.5219727, 1259.0186768
1: -547.6936646, 784.4992065, -538.7338257, 770.7523804, -1316.0861816, 1320.9707031
2: -544.0818481, 773.7381592, -535.1314087, 760.0953979, -1302.3054199, 1307.0290527
3: -674.3672485, 900.7269897, -663.1502686, 884.9095459, -1556.9705811, 1561.6621094
4: -585.6768188, 884.7250366, -576.4880981, 869.2595215, -1451.9720459, 1458.2939453

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6224671, upper bound: 886.6416057
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6473270, upper bound: 886.6457568
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -483.8485718, 783.6398926, -479.0361633, 774.8145142, -1258.6630859, 1262.5919189
1: -547.6936646, 784.4992065, -542.4277954, 773.5675659, -1319.6575928, 1324.9451904
2: -544.0818481, 773.7381592, -539.1839600, 763.2622681, -1306.3403320, 1311.2099609
3: -674.3672485, 900.7269897, -666.4015503, 888.5993042, -1561.5582275, 1565.2165527
4: -585.6768188, 884.7250366, -581.7400513, 873.2807007, -1456.5612793, 1463.8089600

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6224671, upper bound: 886.6629038
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6473270, upper bound: 886.6639985
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -473.6373901, 766.1722412, -333.7850647, 543.7745972, -1017.4119873, 1099.9572754
1: -536.4125366, 764.9110718, -374.2675781, 536.1069336, -1072.5195312, 1139.1787109
2: -533.1043701, 754.7048340, -376.0095520, 530.2505493, -1063.3549805, 1130.7143555
3: -659.0275879, 878.6658936, -458.7491150, 617.5330200, -1276.5605469, 1337.4150391
4: -575.2755737, 863.4638062, -406.9678955, 604.8599243, -1180.1354980, 1270.4316406

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609297, upper bound: 886.6620283
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609297, upper bound: 886.6620924
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -474.0287781, 766.7897339, -341.3853760, 555.9767456, -1030.0054932, 1108.1749268
1: -536.8328857, 765.5568848, -382.8206177, 548.3917236, -1085.2246094, 1148.3773193
2: -533.5804443, 755.3500977, -384.5502319, 542.3837891, -1075.9642334, 1139.9002686
3: -659.5257568, 879.4155884, -469.2127075, 631.5244751, -1291.0500488, 1348.6281738
4: -575.8323975, 864.1560669, -415.9773254, 618.9276733, -1194.7597656, 1280.1330566

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609303, upper bound: 886.6620283
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609303, upper bound: 886.6620924
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -484.4251404, 784.4790039, -333.7850647, 543.7745972, -1028.1997070, 1118.2640381
1: -548.2562866, 783.3593140, -374.2675781, 536.1069336, -1084.3632812, 1157.6267090
2: -545.1821289, 772.9958496, -376.0095520, 530.2505493, -1075.4326172, 1149.0052490
3: -673.9692993, 899.9773560, -458.7491150, 617.5330200, -1291.5023193, 1358.7260742
4: -587.7809448, 884.0230103, -406.9678955, 604.8599243, -1192.6408691, 1290.9909668

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6496669, upper bound: 886.6547570
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6587209, upper bound: 886.6612640
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -484.0476685, 783.9137573, -341.3853760, 555.9767456, -1040.0244141, 1125.2990723
1: -547.8065186, 782.8142700, -382.8206177, 548.3917236, -1096.1981201, 1165.6348877
2: -544.7919312, 772.4473877, -384.5502319, 542.3837891, -1087.1757812, 1156.9975586
3: -673.3983154, 899.4086914, -469.2127075, 631.5244751, -1304.9228516, 1368.6213379
4: -587.4901123, 883.2806396, -415.9773254, 618.9276733, -1206.4177246, 1299.2578125

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505182, upper bound: 886.6558011
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6584286, upper bound: 886.6613450
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -476.7803650, 771.6308594, -1251.8775635, 1253.5504150
1: -543.8629150, 775.5766602, -539.9763184, 772.4961548, -1314.3142090, 1313.9824219
2: -540.5451050, 765.2640991, -536.3087158, 761.8306885, -1300.6263428, 1300.5662842
3: -668.1536255, 890.8625488, -664.6658936, 886.8717041, -1553.0041504, 1554.1018066
4: -583.1776733, 875.6425781, -577.7337036, 871.2992554, -1451.8029785, 1451.0129395

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6499864, upper bound: 886.6401595
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6468968, upper bound: 886.6401612
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -480.2467346, 776.7702637, -1257.0169678, 1257.0169678
1: -543.8629150, 775.5766602, -543.8629150, 775.5766602, -1318.1478271, 1318.1479492
2: -540.5451050, 765.2640991, -540.5451050, 765.2640991, -1304.9301758, 1304.9301758
3: -668.1536255, 890.8625488, -668.1536255, 890.8625488, -1557.8907471, 1557.8907471
4: -583.1776733, 875.6425781, -583.1776733, 875.6425781, -1456.7600098, 1456.7600098

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6499864, upper bound: 886.6588006
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6468968, upper bound: 886.6585555
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -476.7803650, 771.6308594, -1262.1237793, 1271.0093994
1: -555.0930786, 793.1872559, -539.9763184, 772.4961548, -1325.5141602, 1331.7198486
2: -552.0131836, 782.6972656, -536.3087158, 761.8306885, -1312.1339111, 1318.0555420
3: -682.3418579, 911.2104492, -664.6658936, 886.8717041, -1567.2576904, 1574.6436768
4: -595.0556030, 895.1871338, -577.7337036, 871.2992554, -1463.5970459, 1470.5362549

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6494050, upper bound: 886.6474951
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6478311
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -480.2467346, 776.7702637, -1267.2631836, 1274.4758301
1: -555.0930786, 793.1872559, -543.8629150, 775.5766602, -1329.3479004, 1335.8853760
2: -552.0131836, 782.6972656, -540.5451050, 765.2640991, -1316.4377441, 1322.4193115
3: -682.3418579, 911.2104492, -668.1536255, 890.8625488, -1572.1442871, 1578.4326172
4: -595.0556030, 895.1871338, -583.1776733, 875.6425781, -1468.5209961, 1476.2830811

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6494050, upper bound: 886.6641938
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6643921
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -455.0776672, 735.9932861, -335.6234131, 544.3007812, -999.3783569, 1071.6165771
1: -514.1742554, 737.5162354, -375.3912048, 538.5775757, -1052.7518311, 1112.9074707
2: -510.8902893, 727.1378784, -377.2436523, 533.0607910, -1043.9510498, 1104.3815918
3: -635.0394897, 848.2443237, -461.3478699, 620.9017944, -1255.9411621, 1309.5921631
4: -548.4417725, 831.5809326, -406.2480164, 608.3732300, -1156.8148193, 1237.4348145

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6313960, upper bound: 886.6528675
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6268831, upper bound: 886.6466147
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6248056, upper bound: 886.6458460
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -488.3928223, 790.5830078, -334.1728516, 545.0695190, -1033.4624023, 1124.7557373
1: -552.8066406, 791.6104736, -375.0410156, 539.1420898, -1091.9483643, 1166.6512451
2: -549.0991211, 780.7411499, -375.8283386, 532.7131348, -1081.8121338, 1156.5693359
3: -680.7133179, 908.8287354, -460.6825256, 621.1873779, -1301.9006348, 1369.5112305
4: -590.9399414, 892.8865967, -405.6159363, 607.4203491, -1198.3603516, 1298.5025635

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6340724, upper bound: 886.6420334
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6380514, upper bound: 886.6380421
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6385389, upper bound: 886.6385812
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -488.3928223, 790.5830078, -360.7536011, 585.6235962, -1074.0163574, 1151.3366699
1: -552.8066406, 791.6104736, -403.9907837, 578.2063599, -1131.0128174, 1195.6010742
2: -549.0991211, 780.7411499, -405.6423035, 571.9915771, -1121.0906982, 1186.3834229
3: -680.7133179, 908.8287354, -495.7805786, 666.1868286, -1346.8999023, 1404.6092529
4: -590.9399414, 892.8865967, -437.6421204, 653.3969116, -1244.3369141, 1330.5286865

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6340724, upper bound: 886.6545575
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6380514, upper bound: 886.6547067
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6385389, upper bound: 886.6562598
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -455.0776672, 735.9932861, -455.1325684, 737.8056030, -1192.5428467, 1190.9764404
1: -514.1742554, 737.5162354, -515.2052612, 737.3058472, -1249.9068604, 1250.3874512
2: -510.8902893, 727.1378784, -512.3187866, 727.2951050, -1236.8270264, 1237.5523682
3: -635.0394897, 848.2443237, -633.8472900, 847.3088379, -1481.0296631, 1480.4094238
4: -548.4417725, 831.5809326, -552.1796265, 831.6014404, -1377.1140137, 1380.5867920

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6130944, upper bound: 886.6435227
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6130944, upper bound: 886.6485917
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -488.3928223, 790.5830078, -489.3033142, 792.2821655, -1280.6750488, 1279.8433838
1: -552.8066406, 791.6104736, -553.7553711, 791.2332153, -1342.6618652, 1343.4537354
2: -549.0991211, 780.7411499, -550.6660767, 780.7595215, -1328.8483887, 1329.7652588
3: -680.7133179, 908.8287354, -680.7116089, 908.9881592, -1588.6220703, 1587.8941650
4: -590.9399414, 892.8865967, -593.5950317, 892.9833374, -1481.5089111, 1483.7514648

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6332117, upper bound: 886.6508530
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6332117, upper bound: 886.6536694
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -473.6373901, 766.1722412, -353.8612061, 574.6835327, -1048.3209229, 1120.0333252
1: -536.4125366, 764.9110718, -396.1875305, 567.2429810, -1103.6555176, 1161.0986328
2: -533.1043701, 754.7048340, -397.9522400, 561.2069092, -1094.3112793, 1152.6568604
3: -659.0275879, 878.6658936, -486.1527710, 653.6115723, -1312.6391602, 1364.8186035
4: -575.2755737, 863.4638062, -429.6215820, 640.9849854, -1216.2604980, 1293.0853271

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6584583, upper bound: 886.6605482
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6584583, upper bound: 886.6620385
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -474.0287781, 766.7897339, -356.0942078, 578.1146240, -1052.1434326, 1122.8839111
1: -536.8328857, 765.5568848, -398.7054749, 570.7372437, -1107.5700684, 1164.2619629
2: -533.5804443, 755.3500977, -400.4403687, 564.6190796, -1098.1992188, 1155.7904053
3: -659.5257568, 879.4155884, -489.3797607, 657.6079712, -1317.1335449, 1368.7952881
4: -575.8323975, 864.1560669, -432.2026978, 644.9401245, -1220.7724609, 1296.3587646

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609295, upper bound: 886.6619570
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6609295, upper bound: 886.6625907
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -484.4251404, 784.4790039, -353.8612061, 574.6835327, -1059.1086426, 1138.3402100
1: -548.2562866, 783.3593140, -396.1875305, 567.2429810, -1115.4992676, 1179.5468750
2: -545.1821289, 772.9958496, -397.9522400, 561.2069092, -1106.3889160, 1170.9478760
3: -673.9692993, 899.9773560, -486.1527710, 653.6115723, -1327.5808105, 1386.1297607
4: -587.7809448, 884.0230103, -429.6215820, 640.9849854, -1228.7657471, 1313.6445312

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594615, upper bound: 886.6613132
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592879, upper bound: 886.6612735
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -484.0476685, 783.9137573, -356.0942078, 578.1146240, -1062.1622314, 1140.0079346
1: -547.8065186, 782.8142700, -398.7054749, 570.7372437, -1118.5437012, 1181.5197754
2: -544.7919312, 772.4473877, -400.4403687, 564.6190796, -1109.4108887, 1172.8876953
3: -673.3983154, 899.4086914, -489.3797607, 657.6079712, -1331.0063477, 1388.7884521
4: -587.4901123, 883.2806396, -432.2026978, 644.9401245, -1232.4301758, 1315.4833984

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6514709, upper bound: 886.6582369
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6585274, upper bound: 886.6616296
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -482.2491760, 781.0880127, -1261.2409668, 1259.0192871
1: -543.8629150, 775.5766602, -545.8766479, 781.9828491, -1323.8479004, 1319.8708496
2: -540.5451050, 765.2640991, -542.2550659, 771.2410278, -1310.0592041, 1306.5642090
3: -668.1536255, 890.8625488, -672.1783447, 897.8613892, -1564.0788574, 1561.6435547
4: -583.1776733, 875.6425781, -583.7127075, 881.8229980, -1462.3474121, 1457.0318604

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6573288, upper bound: 886.6518735
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6579037, upper bound: 886.6522456
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -480.2467346, 776.7702637, -490.0798035, 793.5619507, -1273.8085938, 1266.8497314
1: -543.8629150, 775.5766602, -554.6266479, 792.5357666, -1335.2331543, 1328.8839111
2: -540.5451050, 765.2640991, -551.5410156, 782.0499878, -1321.7702637, 1315.9733887
3: -668.1536255, 890.8625488, -681.7785034, 910.4652100, -1577.6870117, 1571.5827637
4: -583.1776733, 875.6425781, -594.5493164, 894.4362183, -1475.5357666, 1468.0471191

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6573288, upper bound: 886.6567663
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6468968, upper bound: 886.6567333
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -482.2491760, 781.0880127, -1271.5810547, 1276.4782715
1: -555.0930786, 793.1872559, -545.8766479, 781.9828491, -1335.2092285, 1337.7501221
2: -552.0131836, 782.6972656, -542.2550659, 771.2410278, -1321.6362305, 1324.0745850
3: -682.3418579, 911.2104492, -672.1783447, 897.8613892, -1578.6560059, 1582.5148926
4: -595.0556030, 895.1871338, -583.7127075, 881.8229980, -1474.1414795, 1476.5550537

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6557280, upper bound: 886.6555796
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6493643, upper bound: 886.6556660
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -490.4929810, 794.2290649, -490.0798035, 793.5619507, -1284.0548096, 1284.3087158
1: -555.0930786, 793.1872559, -554.6266479, 792.5357666, -1346.5607910, 1346.7425537
2: -552.0131836, 782.6972656, -551.5410156, 782.0499878, -1333.2989502, 1333.4836426
3: -682.3418579, 911.2104492, -681.7785034, 910.4652100, -1592.2266846, 1592.4080811
4: -595.0556030, 895.1871338, -594.5493164, 894.4362183, -1487.2967529, 1487.5701904

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6416579, upper bound: 886.6481032
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6424808, upper bound: 886.6584661
time: 0.94 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.75 seconds
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6518165, upper bound: 886.6450779
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6515394, upper bound: 886.6457548
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6515145, upper bound: 886.6441890
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6512756, upper bound: 886.6445389
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6645052, upper bound: 886.6624638
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6623632, upper bound: 886.6614060
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6645052, upper bound: 886.6631277
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6623632, upper bound: 886.6622589
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6628161, upper bound: 886.6609012
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609566, upper bound: 886.6609108
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6628161, upper bound: 886.6618703
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609566, upper bound: 886.6619908
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6599447, upper bound: 886.6564801
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6599447, upper bound: 886.6594427
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6611485, upper bound: 886.6617156
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6611485, upper bound: 886.6653452
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6601859, upper bound: 886.6565223
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6601859, upper bound: 886.6580942
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6613888, upper bound: 886.6613888
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6613888, upper bound: 886.6631030
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6650601, upper bound: 886.6624638
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6651941, upper bound: 886.6624736
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6732964, upper bound: 886.6709525
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6716239, upper bound: 886.6698204
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6560815, upper bound: 886.6575616
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6619908, upper bound: 886.6609108
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6560815, upper bound: 886.6575616
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6619908, upper bound: 886.6621163
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6448459
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6548275, upper bound: 886.6451358
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6550992, upper bound: 886.6524587
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6548275, upper bound: 886.6528114
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6591596, upper bound: 886.6579273
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6612640, upper bound: 886.6587209
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6591596, upper bound: 886.6583471
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6612640, upper bound: 886.6587209
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6618128, upper bound: 886.6590792
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6612420, upper bound: 886.6590786
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6618128, upper bound: 886.6590792
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6612420, upper bound: 886.6590786
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6432996, upper bound: 886.6457827
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6460482, upper bound: 886.6462344
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6561349, upper bound: 886.6562386
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6561349, upper bound: 886.6562386
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6501756
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6501756
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6527202
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6591291, upper bound: 886.6528374
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6608412, upper bound: 886.6579776
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6612236, upper bound: 886.6586617
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6608412, upper bound: 886.6583763
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6612236, upper bound: 886.6586617
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6620156, upper bound: 886.6582093
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6619850, upper bound: 886.6586120
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6620156, upper bound: 886.6584902
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6619850, upper bound: 886.6586120
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6594218, upper bound: 886.6652353
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6594218, upper bound: 886.6656018
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6597438, upper bound: 886.6652353
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6597438, upper bound: 886.6656019
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6224671, upper bound: 886.6416057
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6473270, upper bound: 886.6457568
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6224671, upper bound: 886.6629038
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6473270, upper bound: 886.6639985
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609297, upper bound: 886.6620283
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609297, upper bound: 886.6620924
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609303, upper bound: 886.6620283
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609303, upper bound: 886.6620924
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6496669, upper bound: 886.6547570
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6587209, upper bound: 886.6612640
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6505182, upper bound: 886.6558011
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6584286, upper bound: 886.6613450
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6499864, upper bound: 886.6401595
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6468968, upper bound: 886.6401612
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6499864, upper bound: 886.6588006
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6468968, upper bound: 886.6585555
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6494050, upper bound: 886.6474951
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6478311
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6494050, upper bound: 886.6641938
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6509901, upper bound: 886.6643921
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6268831, upper bound: 886.6466147
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6248056, upper bound: 886.6458460
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6380514, upper bound: 886.6380421
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6385389, upper bound: 886.6385812
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6380514, upper bound: 886.6547067
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6385389, upper bound: 886.6562598
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6130944, upper bound: 886.6435227
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6130944, upper bound: 886.6485917
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6332117, upper bound: 886.6508530
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6332117, upper bound: 886.6536694
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6584583, upper bound: 886.6605482
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6584583, upper bound: 886.6620385
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609295, upper bound: 886.6619570
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6609295, upper bound: 886.6625907
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6594615, upper bound: 886.6613132
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6592879, upper bound: 886.6612735
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6514709, upper bound: 886.6582369
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6585274, upper bound: 886.6616296
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6573288, upper bound: 886.6518735
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6579037, upper bound: 886.6522456
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6573288, upper bound: 886.6567663
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6468968, upper bound: 886.6567333
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6557280, upper bound: 886.6555796
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6493643, upper bound: 886.6556660
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6416579, upper bound: 886.6481032
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 0, lower bound: -886.6424808, upper bound: 886.6584661

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -312.6366882, 509.3114624, -296.2059021, 484.3117371, -796.9484253, 805.5173340
1: -350.3283081, 502.1779480, -332.4775391, 477.9424438, -828.2707520, 834.6555176
2: -352.0532227, 496.8777161, -333.3239746, 472.3874512, -824.4406738, 830.2016602
3: -429.6225586, 578.4960938, -408.3214722, 550.7556152, -980.3780518, 986.8175659
4: -380.3330078, 566.7119751, -360.5585632, 538.1074829, -918.4404907, 927.2705078

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513267, upper bound: 886.6450779
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513267, upper bound: 886.6450779
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -316.6052551, 515.8109131, -296.2059021, 484.3117371, -800.9167480, 812.0168457
1: -354.7880859, 508.5686035, -332.4775391, 477.9424438, -832.7305298, 841.0460815
2: -356.5292664, 503.2153931, -333.3239746, 472.3874512, -828.9166260, 836.5393677
3: -435.0425415, 585.8359375, -408.3214722, 550.7556152, -985.7981567, 994.1574097
4: -385.1113281, 573.9063721, -360.5585632, 538.1074829, -923.2188110, 934.4649658

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513267, upper bound: 886.6457548
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513267, upper bound: 886.6457548
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -313.7584839, 511.7609253, -303.8962708, 497.0166931, -810.7751465, 815.6572266
1: -351.9067688, 504.8837280, -341.6340332, 491.1337891, -843.0404053, 846.5177612
2: -353.4280090, 499.0716553, -342.2334290, 485.0921326, -838.5200195, 841.3050537
3: -431.4207458, 581.4021606, -419.1253052, 565.8661499, -997.2868652, 1000.5274658
4: -382.4007568, 569.2222900, -370.8247070, 552.6992188, -935.0999756, 940.0469971

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6510689, upper bound: 886.6441890
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6510689, upper bound: 886.6441890
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -317.5999756, 518.0228271, -303.8962708, 497.0166931, -814.6166992, 821.9190674
1: -356.2250671, 511.0173645, -341.6340332, 491.1337891, -847.3587646, 852.6513672
2: -357.7534180, 505.1675720, -342.2334290, 485.0921326, -842.8453979, 847.4010010
3: -436.6629944, 588.4628296, -419.1253052, 565.8661499, -1002.5289917, 1007.5881348
4: -387.0240784, 576.1408081, -370.8247070, 552.6992188, -939.7232666, 946.9655151

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6510689, upper bound: 886.6445389
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6510689, upper bound: 886.6445389
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -325.4639893, 531.0132446, -333.0894165, 543.2191162, -868.6831055, 864.1024780
1: -365.0609131, 523.0159302, -373.7608643, 535.4633789, -900.5242920, 896.7767944
2: -366.6881714, 517.3336182, -375.1878662, 529.5432129, -896.2313843, 892.5214844
3: -447.4625854, 602.6466064, -458.1258850, 616.8977051, -1064.3601074, 1060.7722168
4: -397.1832886, 589.7128906, -406.4069214, 604.0202637, -1001.2034912, 996.1196899

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646735, upper bound: 886.6624638
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646735, upper bound: 886.6624638
time: 0.79 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.73 + 418.25 = 420.99 seconds
