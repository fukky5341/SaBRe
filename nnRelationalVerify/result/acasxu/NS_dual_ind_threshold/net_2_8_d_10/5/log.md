## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 10002.246664433122


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5663.9658203, 5953.3359375, -5663.9658203, 5953.3359375, -11617.3007812, 11617.3007812)
1: (-599.4065552, 432.4301147, -599.4065552, 432.4301147, -1031.8366699, 1031.8366699)
2: (-973.9396973, 1117.4992676, -973.9396973, 1117.4992676, -2091.4389648, 2091.4389648)
3: (-1112.3724365, 707.8311768, -1112.3724365, 707.8311768, -1820.2034912, 1820.2034912)
4: (-842.6556396, 907.6983032, -842.6556396, 907.6983032, -1750.3540039, 1750.3540039)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.37 + 2.12 = 5.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -10002.3466879, upper bound: 10002.3466879

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3438687, upper bound: 10002.3440302
time: 0.84 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3455401, upper bound: 10002.3455400
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.97 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 0, lower bound: -10002.3438687, upper bound: 10002.3440302
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 0, lower bound: -10002.3455401, upper bound: 10002.3455400

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5034.1801758, 5338.9794922, -5354.1748047, 5644.6337891, -10678.8115234, 10693.1542969
1: -535.7406006, 384.8138733, -567.5372314, 409.1065369, -944.8470459, 952.3510132
2: -863.8967285, 1001.3222046, -920.1129150, 1060.0047607, -1923.9011230, 1921.4350586
3: -983.5880737, 631.8367310, -1050.6031494, 670.0122681, -1653.6003418, 1682.4399414
4: -744.6766968, 813.3613281, -795.6074829, 860.8105469, -1605.4870605, 1608.9687500

Time for backsubstitution: 3.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3364405, upper bound: 10002.3362923
time: 0.83 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3381221, upper bound: 10002.3379403
time: 0.86 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5446.1494141, 5731.5761719, -5508.7939453, 5794.0600586, -11240.2089844, 11240.3701172
1: -576.6632690, 415.8731689, -583.0982056, 420.6202698, -997.2835083, 998.9713745
2: -936.3096313, 1076.3522949, -947.1294556, 1088.0214844, -2024.3310547, 2023.4815674
3: -1069.5169678, 680.7265625, -1081.9287109, 688.4092407, -1757.9262695, 1762.6552734
4: -809.8715820, 874.1198120, -819.3659668, 883.6159058, -1693.4875488, 1693.4854736

Time for backsubstitution: 3.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3364405, upper bound: 10002.3402298
time: 0.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3381221, upper bound: 10002.3404154
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 0, lower bound: -10002.3364405, upper bound: 10002.3362923
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 0, lower bound: -10002.3381221, upper bound: 10002.3379403
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 0, lower bound: -10002.3364405, upper bound: 10002.3402298
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 0, lower bound: -10002.3381221, upper bound: 10002.3404154

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4802.4853516, 5099.2514648, -4918.8574219, 5167.1923828, -9969.6777344, 10018.1093750
1: -511.4343567, 367.2150574, -520.8040771, 375.4917908, -886.9261475, 888.0189819
2: -824.5222778, 955.5121460, -846.5653076, 968.6315308, -1793.1534424, 1802.0773926
3: -939.4656372, 603.5291748, -967.5892334, 615.1638184, -1554.6293945, 1571.1184082
4: -711.2980347, 776.0798950, -733.3568115, 786.0254517, -1497.3234863, 1509.4365234

Time for backsubstitution: 3.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.2077394, upper bound: 10002.2232769
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.2097483, upper bound: 10002.2117257
time: 0.71 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5006.5317383, 5309.5307617, -5226.3486328, 5516.1079102, -10522.6396484, 10535.8789062
1: -532.8555298, 382.6975098, -554.6207275, 399.6027222, -932.4581299, 937.3180542
2: -859.1364746, 996.0444946, -896.4974976, 1036.3696289, -1895.5056152, 1892.5418701
3: -978.2245483, 628.3549194, -1022.5795898, 654.2227783, -1632.4472656, 1650.9345703
4: -740.6459961, 809.0112915, -774.5474243, 841.4962158, -1582.1420898, 1583.5582275

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3338851, upper bound: 10002.3349278
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3338851, upper bound: 10002.3379403
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5203.6049805, 5473.3232422, -5062.8017578, 5321.3066406, -10524.9121094, 10536.1250000
1: -551.2360840, 397.2917786, -536.1931152, 386.1746216, -937.4106445, 933.4848633
2: -895.1666870, 1027.1367188, -871.9276733, 996.8685303, -1892.0351562, 1899.0644531
3: -1022.7135620, 650.7227173, -996.3354492, 633.2485352, -1655.9617920, 1647.0578613
4: -774.6868896, 834.0340576, -755.2133179, 809.5620728, -1584.2486572, 1589.2473145

Time for backsubstitution: 3.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2306234, upper bound: 10002.2672450
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2291567, upper bound: 10002.2470912
time: 0.76 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5419.5673828, 5705.6728516, -5379.7656250, 5667.2358398, -11086.8027344, 11085.4375000
1: -573.8903198, 413.9168396, -570.0048828, 411.1446838, -985.0350342, 983.9216919
2: -931.4687500, 1071.5307617, -923.3636475, 1064.5194092, -1995.9881592, 1994.8944092
3: -1063.7447510, 677.4525757, -1053.5222168, 672.5779419, -1736.3226318, 1730.9747314
4: -805.5058594, 870.2040405, -797.9884033, 864.4938354, -1669.9993896, 1668.1920166

Time for backsubstitution: 3.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3398037, upper bound: 10002.3400436
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3398037, upper bound: 10002.3404154
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.03 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.2077394, upper bound: 10002.2232769
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.2097483, upper bound: 10002.2117257
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.3338851, upper bound: 10002.3349278
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.3338851, upper bound: 10002.3379403
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.2306234, upper bound: 10002.2672450
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.2291567, upper bound: 10002.2470912
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.3398037, upper bound: 10002.3400436
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.03
Output dim: 0, lower bound: -10002.3398037, upper bound: 10002.3404154

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4677.9545898, 4959.0332031, -5226.3486328, 5516.1079102, -10194.0625000, 10185.3818359
1: -498.2456360, 357.4674377, -554.6207275, 399.6027222, -897.8482666, 912.0881348
2: -803.8273926, 928.6170654, -896.4974976, 1036.3696289, -1840.1970215, 1825.1145020
3: -916.5253906, 588.2615356, -1022.5795898, 654.2227783, -1570.7479248, 1610.8410645
4: -693.8519287, 754.0755005, -774.5474243, 841.4962158, -1535.3477783, 1528.6226807

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1802518, upper bound: 10002.0984891
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1332878, upper bound: 10002.0983134
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4909.5312500, 5205.8486328, -5226.3486328, 5516.1079102, -10425.6386719, 10432.1972656
1: -522.6376343, 375.1992493, -554.6207275, 399.6027222, -922.2402344, 929.8199463
2: -842.4616699, 977.3558960, -896.4974976, 1036.3696289, -1878.8311768, 1873.8533936
3: -959.6598511, 616.1436768, -1022.5795898, 654.2227783, -1613.8825684, 1638.7232666
4: -726.8801880, 793.7933960, -774.5474243, 841.4962158, -1568.3759766, 1568.3406982

Time for backsubstitution: 3.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1802518, upper bound: 10002.2463269
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1332878, upper bound: 10002.2390928
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5001.1386719, 5265.8598633, -4949.9770508, 5208.5610352, -10209.6972656, 10215.8349609
1: -530.0511475, 381.6755371, -524.5846558, 377.5044250, -907.5555420, 906.2601929
2: -860.6174927, 988.2286377, -852.6024170, 975.6636353, -1836.2811279, 1840.8306885
3: -983.6881714, 625.7975464, -974.2035522, 619.5351562, -1603.2233887, 1600.0010986
4: -744.8390503, 802.0780029, -738.1619873, 792.3303223, -1537.1693115, 1540.2399902

Time for backsubstitution: 3.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0984891, upper bound: 10002.1802517
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.0984891, upper bound: 10002.2672450
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5300.3881836, 5597.2622070, -4911.1157227, 5165.0649414, -10465.4531250, 10508.3750000
1: -563.7831421, 404.3563232, -520.7536621, 374.7460632, -938.5291748, 925.1099243
2: -913.9268188, 1049.7193604, -846.1000977, 968.3903198, -1882.3171387, 1895.8192139
3: -1047.2927246, 665.0119629, -967.8665771, 614.5764160, -1661.8688965, 1632.8784180
4: -792.6663208, 852.4141846, -733.4279175, 786.0725708, -1578.7386475, 1585.8420410

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0983135, upper bound: 10002.1575552
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0983135, upper bound: 10002.2380545
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5011.0874023, 5270.4873047, -5379.7656250, 5667.2358398, -10678.3232422, 10650.2509766
1: -530.9620972, 382.2428589, -570.0048828, 411.1446838, -942.1068115, 952.2476807
2: -863.0095215, 987.3484497, -923.3636475, 1064.5194092, -1927.5289307, 1910.7121582
3: -986.1509399, 626.9854126, -1053.5222168, 672.5779419, -1658.7286377, 1680.5073242
4: -747.3746338, 801.8314209, -797.9884033, 864.4938354, -1611.8682861, 1599.8195801

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2672450, upper bound: 10002.2306233
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.2239231, upper bound: 10002.2291566
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5316.4672852, 5604.1386719, -5379.7656250, 5667.2358398, -10983.7031250, 10983.9033203
1: -563.5587158, 406.3412170, -570.0048828, 411.1446838, -974.7033691, 976.3460083
2: -912.4025269, 1052.7518311, -923.3636475, 1064.5194092, -1976.9218750, 1976.1154785
3: -1040.9370117, 664.8493042, -1053.5222168, 672.5779419, -1713.5148926, 1718.3715820
4: -788.3790283, 854.9104614, -797.9884033, 864.4938354, -1652.8724365, 1652.8984375

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2672450, upper bound: 10002.2637527
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2239231, upper bound: 10002.2558034
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.76 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.1802518, upper bound: 10002.0984891
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.1332878, upper bound: 10002.0983134
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.1802518, upper bound: 10002.2463269
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.1332878, upper bound: 10002.2390928
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.0984891, upper bound: 10002.1802517
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.0984891, upper bound: 10002.2672450
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.0983135, upper bound: 10002.1575552
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.0983135, upper bound: 10002.2380545
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.2672450, upper bound: 10002.2306233
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.2239231, upper bound: 10002.2291566
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.2672450, upper bound: 10002.2637527
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.76
Output dim: 0, lower bound: -10002.2239231, upper bound: 10002.2558034

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5001.1386719, 5265.8598633, -4901.3833008, 5160.8613281, -10161.9990234, 10167.2431641
1: -530.0511475, 381.6755371, -519.6757812, 373.8032227, -903.8541870, 901.3513184
2: -860.6174927, 988.2286377, -844.2077026, 966.7232056, -1827.3405762, 1832.4360352
3: -983.6881714, 625.7975464, -964.6110229, 613.6521606, -1597.3402100, 1590.4085693
4: -744.8390503, 802.0780029, -730.7754517, 785.0710449, -1529.9100342, 1532.8532715

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.0976353, upper bound: 10002.2672450
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.0976353, upper bound: 10002.2672450
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4901.3833008, 5160.8613281, -5153.6157227, 5429.1738281, -10330.5566406, 10314.4765625
1: -519.6757812, 373.8032227, -546.0700073, 393.8429565, -913.5187378, 919.8731079
2: -844.2077026, 966.7232056, -884.7340698, 1020.5137329, -1864.7209473, 1851.4571533
3: -964.6110229, 613.6521606, -1010.2072754, 644.3454590, -1608.9565430, 1623.8592529
4: -730.7754517, 785.0710449, -765.1389160, 828.2041626, -1558.9792480, 1550.2098389

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0844444, upper bound: 10002.0926244
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2616749, upper bound: 10002.2204610
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5188.8916016, 5468.4423828, -5153.6157227, 5429.1738281, -10618.0654297, 10622.0585938
1: -549.8986206, 396.5855408, -546.0700073, 393.8429565, -943.7415771, 942.6555176
2: -890.5813599, 1027.7204590, -884.7340698, 1020.5137329, -1911.0948486, 1912.4545898
3: -1016.4488525, 648.8045654, -1010.2072754, 644.3454590, -1660.7943115, 1659.0118408
4: -769.8134155, 834.2540283, -765.1389160, 828.2041626, -1598.0175781, 1599.3929443

Time for backsubstitution: 3.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.2227408, upper bound: 10001.2716627
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0590778, upper bound: 10001.8975482
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3039579, upper bound: 10002.2564431
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5176.9467773, 5463.2568359, -5449.5849609, 5760.4526367, -10937.3994141, 10912.8417969
1: -549.5021973, 395.6229858, -579.3129272, 416.1517944, -965.6539917, 974.9359131
2: -888.4650879, 1026.7276611, -937.6220093, 1081.8193359, -1970.2844238, 1964.3492432
3: -1014.5339966, 647.8764648, -1073.3809814, 683.4273071, -1697.9613037, 1721.2568359
4: -768.1171875, 833.7069092, -812.2241821, 878.2297363, -1646.3468018, 1645.9306641

Time for backsubstitution: 3.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1635611, upper bound: 10002.2123237
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1617271, upper bound: 10002.1587649
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.59 seconds
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.0976353, upper bound: 10002.2672450
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.0976353, upper bound: 10002.2672450
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.0844444, upper bound: 10002.0926244
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.2616749, upper bound: 10002.2204610
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.0590778, upper bound: 10001.8975482
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.3039579, upper bound: 10002.2564431
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.1635611, upper bound: 10002.2123237
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.59
Output dim: 0, lower bound: -10002.1617271, upper bound: 10002.1587649

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4818.3666992, 5078.5073242, -4901.3833008, 5160.8613281, -9979.2255859, 9979.8906250
1: -511.2669678, 367.4244080, -519.6757812, 373.8032227, -885.0700073, 887.1002197
2: -829.9871826, 951.2772217, -844.2077026, 966.7232056, -1796.7100830, 1795.4846191
3: -948.4185181, 603.6028442, -964.6110229, 613.6521606, -1562.0705566, 1568.2138672
4: -718.2967529, 772.5512695, -730.7754517, 785.0710449, -1503.3677979, 1503.3264160

Time for backsubstitution: 3.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0911844, upper bound: 10002.0559112
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2204610, upper bound: 10002.2616749
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5091.1479492, 5367.0122070, -4901.3833008, 5160.8613281, -10252.0097656, 10268.3955078
1: -539.7150879, 389.1088562, -519.6757812, 373.8032227, -913.5182495, 908.7846069
2: -873.9039307, 1008.9127197, -844.2077026, 966.7232056, -1840.6268311, 1853.1203613
3: -997.7485962, 636.7312622, -964.6110229, 613.6521606, -1611.4005127, 1601.3422852
4: -755.6251221, 818.7530518, -730.7754517, 785.0710449, -1540.6960449, 1549.5284424

Time for backsubstitution: 3.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0911844, upper bound: 10002.0559116
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2204610, upper bound: 10002.2616749
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4841.9462891, 5102.4360352, -5135.5766602, 5411.4008789, -10253.3466797, 10238.0126953
1: -513.6549683, 369.3138123, -544.2543945, 392.4729614, -906.1279297, 913.5679932
2: -833.9235840, 955.8413086, -881.6256104, 1017.2082520, -1851.1317139, 1837.4669189
3: -952.9039307, 606.4853516, -1006.6418457, 642.1689453, -1595.0728760, 1613.1268311
4: -721.7682495, 776.2304077, -762.4174805, 825.5208130, -1547.2888184, 1538.6479492

Time for backsubstitution: 3.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.2190166, upper bound: 10002.2007773
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.2190166, upper bound: 10002.2201931
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5129.5620117, 5409.9902344, -5135.5766602, 5411.4008789, -10540.9619141, 10545.5664062
1: -543.9359131, 392.0819702, -544.2543945, 392.4729614, -936.4088745, 936.3362427
2: -880.3242798, 1016.8628540, -881.6256104, 1017.2082520, -1897.5321045, 1898.4885254
3: -1004.6647339, 641.6524048, -1006.6418457, 642.1689453, -1646.8337402, 1648.2941895
4: -760.8099365, 825.4412842, -762.4174805, 825.5208130, -1586.3306885, 1587.8585205

Time for backsubstitution: 3.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2992972, upper bound: 10002.2487082
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2992972, upper bound: 10002.2554016
time: 0.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.66 seconds
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.0911844, upper bound: 10002.0559112
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.2204610, upper bound: 10002.2616749
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.0911844, upper bound: 10002.0559116
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.2204610, upper bound: 10002.2616749
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.2190166, upper bound: 10002.2007773
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.2190166, upper bound: 10002.2201931
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.2992972, upper bound: 10002.2487082
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 0, lower bound: -10002.2992972, upper bound: 10002.2554016

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4799.2172852, 5059.7265625, -4841.9462891, 5102.4360352, -9901.6523438, 9901.6728516
1: -509.3295593, 365.9782104, -513.6549683, 369.3138123, -878.6431885, 879.6331787
2: -826.6784058, 947.7780762, -833.9235840, 955.8413086, -1782.5195312, 1781.7016602
3: -944.6529541, 601.2952881, -952.9039307, 606.4853516, -1551.1381836, 1554.1992188
4: -715.3991699, 769.7080078, -721.7682495, 776.2304077, -1491.6296387, 1491.4761963

Time for backsubstitution: 3.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1630599, upper bound: 10002.2395334
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1634764, upper bound: 10002.2414294
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5072.6586914, 5348.8520508, -4841.9462891, 5102.4360352, -10175.0947266, 10190.7978516
1: -537.8611450, 387.7083740, -513.6549683, 369.3138123, -907.1747437, 901.3633423
2: -870.6982422, 1005.5376587, -833.9235840, 955.8413086, -1826.5391846, 1839.4611816
3: -994.0570679, 634.5073242, -952.9039307, 606.4853516, -1600.5419922, 1587.4112549
4: -752.8031006, 816.0126343, -721.7682495, 776.2304077, -1529.0334473, 1537.7808838

Time for backsubstitution: 3.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1604176, upper bound: 10002.2107659
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1611923, upper bound: 10002.2073963
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5129.5620117, 5409.9902344, -4649.8081055, 4939.3471680, -10068.9091797, 10059.7988281
1: -543.9359131, 392.0819702, -495.7372742, 355.1924744, -899.1284180, 887.8191528
2: -880.3242798, 1016.8628540, -798.4520264, 927.5086670, -1807.8325195, 1815.3145752
3: -1004.6647339, 641.6524048, -910.4073486, 584.1403198, -1588.8050537, 1552.0598145
4: -760.8099365, 825.4412842, -689.2429199, 752.7982178, -1513.6081543, 1514.6839600

Time for backsubstitution: 3.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0650763, upper bound: 10001.9175193
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0997275, upper bound: 10001.8712096
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5129.5620117, 5409.9902344, -5072.6586914, 5348.8520508, -10478.4130859, 10482.6484375
1: -543.9359131, 392.0819702, -537.8611450, 387.7083740, -931.6442871, 929.9429932
2: -880.3242798, 1016.8628540, -870.6982422, 1005.5376587, -1885.8615723, 1887.5606689
3: -1004.6647339, 641.6524048, -994.0570679, 634.5073242, -1639.1721191, 1635.7094727
4: -760.8099365, 825.4412842, -752.8031006, 816.0126343, -1576.8225098, 1578.2442627

Time for backsubstitution: 3.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0650763, upper bound: 10001.9175193
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0997275, upper bound: 10001.8712096
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.55 seconds
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.1630599, upper bound: 10002.2395334
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.1634764, upper bound: 10002.2414294
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.1604176, upper bound: 10002.2107659
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.1611923, upper bound: 10002.2073963
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.0650763, upper bound: 10001.9175193
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.0997275, upper bound: 10001.8712096
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.0650763, upper bound: 10001.9175193
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.55
Output dim: 0, lower bound: -10002.0997275, upper bound: 10001.8712096

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.50 + 140.73 = 146.22 seconds
