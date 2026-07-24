## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 9670.419019151372


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4668.2700195, 5783.0693359, -4668.2700195, 5783.0693359, -10451.3398438, 10451.3398438)
1: (-543.5577393, 490.9529724, -543.5577393, 490.9529724, -1034.5106201, 1034.5107422)
2: (-317.8979187, 543.0774536, -317.8979187, 543.0774536, -860.9753418, 860.9753418)
3: (-261.9543152, 561.4524536, -261.9543152, 561.4524536, -823.4067383, 823.4067383)
4: (-378.8520508, 479.5419617, -378.8520508, 479.5419617, -858.3939209, 858.3939209)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.66 + 1.98 = 4.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6124314

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6112676
time: 1.00 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6124314
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.93 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.93
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6112676
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.93
Output dim: 0, lower bound: -9670.6124314, upper bound: 9670.6124314

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4433.1440430, 5540.1186523, -4509.0869141, 5619.7031250, -10052.8457031, 10049.2050781
1: -521.1546631, 468.2802429, -528.4963379, 475.6590271, -996.8136597, 996.7766113
2: -303.3691406, 519.1791382, -308.1034851, 526.9620972, -830.3312378, 827.2825317
3: -249.1298370, 537.3625488, -253.2921143, 545.2320557, -794.3618774, 790.6546631
4: -361.3183289, 458.1906433, -367.0343323, 465.1442566, -826.4625244, 825.2249756

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6103666, upper bound: 9670.6079618
time: 0.64 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
time: 0.88 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6122238, upper bound: 9670.6109909
time: 0.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4669.8085938, 5876.4956055, -4592.3217773, 5703.0927734, -10372.9003906, 10468.8164062
1: -553.6687012, 494.5637512, -536.3544922, 483.4186401, -1037.0874023, 1030.9180908
2: -320.0248718, 550.0695190, -313.0107422, 535.2421265, -855.2669678, 863.0802612
3: -262.5372314, 569.7146606, -257.7565308, 553.6063843, -816.1434937, 827.4710693
4: -381.8187256, 484.5468750, -373.1360474, 472.3986511, -854.2173462, 857.6828613

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6103482, upper bound: 9670.6094685
time: 0.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6094685, upper bound: 9670.6094685
time: 1.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.89 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.89
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.89
Output dim: 0, lower bound: -9670.6122238, upper bound: 9670.6109909
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.89
Output dim: 0, lower bound: -9670.6103482, upper bound: 9670.6094685
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.89
Output dim: 0, lower bound: -9670.6094685, upper bound: 9670.6094685

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4336.0000000, 5444.9736328, -4296.4707031, 5350.3510742, -9686.3505859, 9741.4443359
1: -512.3503418, 459.2509460, -503.1836548, 453.3083496, -965.6586304, 962.4343872
2: -297.4919128, 509.6576233, -293.5616760, 501.9999390, -799.4916992, 803.2192993
3: -243.9003906, 527.8240356, -241.3035583, 519.0382690, -762.9386597, 769.1275635
4: -354.2825928, 449.6995544, -350.0414734, 442.9654236, -797.2479858, 799.7410278

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026354
time: 0.88 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4402.6445312, 5507.6503906, -4447.8300781, 5553.9443359, -9956.5878906, 9955.4804688
1: -518.1906738, 465.2778931, -522.4993896, 469.5838623, -987.7745361, 987.7772827
2: -301.3881226, 516.0007324, -304.1086426, 520.5347900, -821.9227295, 820.1093750
3: -247.4611664, 534.1514893, -249.9271088, 538.7529907, -786.2141113, 784.0785522
4: -359.0041809, 455.3313904, -362.3632507, 459.3629456, -818.3670654, 817.6946411

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6097773
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087375
time: 1.11 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4542.5532227, 5714.3530273, -4229.8784180, 5238.3085938, -9780.8613281, 9944.2314453
1: -538.4603271, 480.8435364, -492.7318420, 444.0748291, -982.5351562, 973.5751343
2: -311.0395203, 534.9917603, -287.2879028, 492.0666809, -803.1060791, 822.2796021
3: -255.4008484, 553.8229980, -237.3279114, 508.0180054, -763.4188232, 791.1508789
4: -371.2046509, 471.1484680, -342.7759094, 434.0765686, -805.2811279, 813.9243774

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6030149, upper bound: 9670.5699626
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6030149, upper bound: 9670.6087643
time: 0.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4585.2680664, 5785.4931641, -5668.4926758, 6653.5283203, -11238.7958984, 11453.9853516
1: -545.1640625, 486.3749390, -624.2644653, 579.0682983, -1124.2324219, 1110.6394043
2: -314.7286377, 541.1809692, -374.0587158, 634.3189697, -949.0476074, 915.2396240
3: -257.9661865, 560.6967773, -315.8566895, 650.6780396, -908.6442261, 876.5534668
4: -375.4327393, 476.7163696, -446.6561279, 562.1845093, -937.6171875, 923.3724976

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6027677, upper bound: 9670.5699626
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087643
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026354
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6097773
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087375
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.6030149, upper bound: 9670.5699626
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.6030149, upper bound: 9670.6087643
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.6027677, upper bound: 9670.5699626
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087643

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3955.3398438, 4964.0791016, -4169.1499023, 5188.4248047, -9143.7646484, 9133.2275391
1: -467.2340698, 418.4709778, -487.9863892, 439.6110229, -906.8450928, 906.4573975
2: -270.8006287, 464.7933960, -284.5913086, 486.9143066, -757.7148438, 749.3847046
3: -222.5997620, 480.6234741, -234.1683197, 503.1446838, -725.7444458, 714.7918091
4: -322.7438354, 409.8748779, -339.4357605, 429.6176453, -752.3614502, 749.3104248

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6014958
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026354
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5285.4497070, 6329.4018555, -4212.8115234, 5266.9335938, -10552.3828125, 10542.2119141
1: -594.0287476, 546.1616821, -495.4900208, 445.4332275, -1039.4619141, 1041.6517334
2: -353.0127563, 600.0656128, -288.4777222, 493.6873169, -846.7000732, 888.5433350
3: -295.7244568, 616.6598511, -236.7640686, 510.7768860, -806.5012207, 853.4239502
4: -421.4283142, 530.7645874, -343.9214478, 435.5381775, -856.9664917, 874.6860352

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6009239
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4026.0649414, 5030.3256836, -4316.9633789, 5387.9916992, -9414.0566406, 9347.2880859
1: -473.4000244, 424.8351746, -506.9384155, 455.5352173, -928.9351196, 931.7735596
2: -274.9015503, 471.4940186, -294.9025269, 505.0600586, -779.9614868, 766.3965454
3: -226.3385315, 487.3182373, -242.5909576, 522.4878540, -748.8264160, 729.9091797
4: -327.7162170, 415.8585510, -351.4770203, 445.6046448, -773.3207397, 767.3355713

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087375
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087375
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5364.1201172, 6397.9194336, -4377.0161133, 5481.5908203, -10845.7109375, 10774.9355469
1: -600.5177002, 552.9967651, -515.7877197, 462.8458557, -1063.3635254, 1068.7844238
2: -357.3586426, 607.2544556, -299.7594604, 513.3993530, -870.7578125, 907.0139160
3: -299.8357239, 623.7329102, -246.0829010, 531.5927124, -831.4284668, 869.8156738
4: -426.6411743, 537.3050537, -357.1365967, 453.0253906, -879.6665039, 894.4416504

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6086913
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4285.2329102, 5402.6528320, -4115.9702148, 5124.7797852, -9410.0126953, 9518.6220703
1: -509.1071167, 454.5322571, -482.1737366, 433.4332275, -942.5401611, 936.7059937
2: -293.8904724, 505.7122498, -280.3710022, 480.7248230, -774.6152954, 786.0831299
3: -241.1091003, 523.3021240, -231.2513275, 496.6408081, -737.7498779, 754.5533447
4: -351.1876831, 445.1669312, -334.5133667, 424.0021057, -775.1896973, 779.6800537

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6029329, upper bound: 9670.5652639
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5484159, upper bound: 9670.3972069
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3331563, upper bound: 9670.3735266
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4487.5913086, 5654.0336914, -4205.0576172, 5210.9482422, -9698.5380859, 9859.0878906
1: -532.9734497, 475.3341370, -490.2413330, 441.5776672, -974.5511475, 965.5754395
2: -307.3937378, 529.1367798, -285.6298218, 489.4136658, -796.8073120, 814.7666016
3: -252.3545074, 547.9310913, -235.9486237, 505.3522339, -757.7067261, 783.8796997
4: -366.9347839, 465.9044495, -340.8461914, 431.6909180, -798.6257324, 806.7506104

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6097468, upper bound: 9670.6087240
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6096901, upper bound: 9670.6087643
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099339, upper bound: 9670.6087643
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4329.1972656, 5478.2265625, -5555.2382812, 6548.8208008, -10878.0156250, 11033.4648438
1: -516.3309937, 460.2940979, -614.5158691, 568.7925415, -1085.1234131, 1074.8099365
2: -297.7427063, 512.2425537, -367.4331360, 623.5089722, -921.2514648, 879.6756592
3: -243.7638550, 530.6349487, -309.7928772, 639.7559204, -883.5197754, 840.4277344
4: -355.5923767, 451.0020142, -438.7652283, 552.3790283, -907.9714355, 889.7672119

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6027677, upper bound: 9670.5652639
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022972, upper bound: 9670.5652639
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4525.8984375, 5721.4047852, -5643.7167969, 6625.5634766, -11151.4619141, 11365.1210938
1: -539.3196411, 480.4938965, -621.7311401, 576.5426025, -1115.8620605, 1102.2247314
2: -310.8395996, 534.9328613, -372.3703918, 631.6258545, -942.4654541, 907.3031616
3: -254.6993866, 554.4069214, -314.4667969, 647.9556274, -902.6549683, 868.8737183
4: -370.8798828, 471.1275940, -444.6862183, 559.7460938, -930.6259766, 915.8137817

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087240
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
time: 0.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.70 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6014958
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026354
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6009239
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087375
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087375
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6086913
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.5484159, upper bound: 9670.3972069
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.3331563, upper bound: 9670.3735266
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6096901, upper bound: 9670.6087643
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6099339, upper bound: 9670.6087643
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6027677, upper bound: 9670.5652639
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6022972, upper bound: 9670.5652639
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6087240
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3955.3398438, 4964.0791016, -3930.9780273, 4883.7163086, -8839.0556641, 8895.0527344
1: -467.2340698, 418.4709778, -459.3248901, 413.8519287, -881.0859985, 877.7958984
2: -270.8006287, 464.7933960, -267.7149353, 458.5202942, -729.3208008, 732.5082397
3: -222.5997620, 480.6234741, -220.7507629, 473.3298645, -695.9296265, 701.3742676
4: -322.7438354, 409.8748779, -319.5362854, 404.4896851, -727.2335205, 729.4109497

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6025183
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3955.3398438, 4964.0791016, -5228.2700195, 6199.5429688, -10154.8828125, 10192.3476562
1: -467.2340698, 418.4709778, -581.7132568, 537.5338745, -1004.7677612, 1000.1842041
2: -270.8006287, 464.7933960, -347.2168579, 589.5104370, -860.3109741, 812.0101929
3: -222.5997620, 480.6234741, -292.0276184, 605.2127686, -827.8125000, 772.6511230
4: -322.7438354, 409.8748779, -414.8259277, 521.9668579, -844.7106934, 824.7005615

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5259.6264648, 6291.2729492, -4017.9284668, 4986.1933594, -10245.8203125, 10309.2001953
1: -590.3729858, 543.1751709, -468.5183411, 423.2058411, -1013.5787354, 1011.6934814
2: -351.0941162, 596.6195068, -274.3129578, 468.0540161, -819.1481323, 870.9324951
3: -294.2540894, 613.0100098, -225.6781158, 483.8479309, -778.1019897, 838.6881104
4: -419.0887756, 527.8095093, -326.6222839, 413.2391663, -832.3278809, 854.4317627

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.3845135, upper bound: 9670.5051378
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5120.8686523, 6114.1743164, -5839.1850586, 6655.8461914, -11776.7148438, 11953.3593750
1: -573.6341553, 528.4756470, -623.1582031, 586.9758301, -1160.6099854, 1151.6337891
2: -341.6875610, 580.0936890, -379.1402893, 639.3779297, -981.0654297, 959.2340088
3: -286.4080200, 595.9288330, -323.3581848, 655.3825684, -941.7905884, 919.2869873
4: -407.6802063, 513.3256836, -452.9455872, 568.6832886, -976.3634644, 966.2712402

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.3753379, upper bound: 9670.5035236
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4026.0649414, 5030.3256836, -4076.0263672, 5080.2885742, -9106.3535156, 9106.3515625
1: -473.4000244, 424.8351746, -478.0450439, 429.4733582, -902.8734131, 902.8801880
2: -274.9015503, 471.4940186, -277.8355713, 476.4255066, -751.3270264, 749.3295898
3: -226.3385315, 487.3182373, -229.0134277, 492.3004761, -718.6390381, 716.3316650
4: -327.7162170, 415.8585510, -331.3485107, 420.2449036, -747.9609985, 747.2070312

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097773
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4026.0649414, 5030.3256836, -5416.7402344, 6444.3969727, -10470.4619141, 10447.0654297
1: -473.4000244, 424.8351746, -604.8671875, 557.6233521, -1031.0234375, 1029.7023926
2: -274.9015503, 471.4940186, -360.2341919, 612.0624390, -886.9639893, 831.7282104
3: -226.3385315, 487.3182373, -302.6403809, 628.5065918, -854.8450928, 789.9586182
4: -327.7162170, 415.8585510, -430.1502075, 541.6819458, -869.3979492, 846.0087891

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097773
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5338.4321289, 6359.7348633, -4172.5644531, 5190.9453125, -10529.3759766, 10532.2988281
1: -596.8581543, 550.0131836, -487.9042358, 439.7306824, -1036.5888672, 1037.9171143
2: -355.4421082, 603.8078003, -285.0172729, 486.6906433, -842.1327515, 888.8250732
3: -298.3732300, 620.1014404, -234.4779358, 503.6874084, -802.0606689, 854.5793457
4: -424.2994080, 534.3508301, -339.1093445, 429.8288879, -854.1281738, 873.4601440

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6086913
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6086913
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5204.0468750, 6185.4648438, -6002.3598633, 6867.6982422, -12071.7441406, 12187.8222656
1: -580.3825073, 535.6510010, -643.4290771, 604.8945923, -1185.2769775, 1179.0800781
2: -346.2510681, 587.6340942, -390.7414856, 658.8297729, -1005.0808105, 978.3756104
3: -290.7414856, 603.3508301, -332.8655090, 675.3732300, -966.1147461, 936.2163086
4: -413.1523743, 520.1914673, -466.5079041, 585.9307251, -999.0831299, 986.6993408

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6038207
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6050793
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4274.3378906, 5389.2797852, -4038.6643066, 5026.4375000, -9300.7734375, 9427.9443359
1: -507.8391724, 453.4029846, -472.7755432, 425.2343445, -933.0734863, 926.1785278
2: -293.1655884, 504.4255371, -275.1160889, 471.3523254, -764.5178223, 779.5415039
3: -240.5113220, 522.0054932, -226.9558563, 487.0974731, -727.6087646, 748.9613647
4: -350.3010559, 444.0297241, -328.0862427, 415.7737732, -766.0748291, 772.1159058

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3331563, upper bound: 9670.3735266
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9670.3331563, upper bound: 9670.3735266
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4317.9233398, 5452.0322266, -3735.9919434, 4717.1445312, -9035.0683594, 9188.0244141
1: -514.0682983, 457.8484802, -444.5668335, 396.2758179, -910.3439941, 902.4152832
2: -295.9463806, 509.8195496, -256.2567139, 440.8562622, -736.8024292, 766.0762329
3: -242.9787445, 528.0554199, -210.5173798, 456.3711853, -699.3499146, 738.5728149
4: -353.3945007, 448.7382507, -305.4596863, 388.4521484, -741.8465576, 754.1979370

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6096901, upper bound: 9670.6087643
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6096901, upper bound: 9670.6087643
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4470.6425781, 5636.1157227, -4175.6982422, 5179.9926758, -9650.6347656, 9811.8125000
1: -531.3225708, 473.6918945, -487.3818054, 438.7275085, -970.0500488, 961.0737305
2: -306.3218384, 527.3873901, -283.7781677, 486.3761597, -792.6979980, 811.1655273
3: -251.4130859, 546.1622314, -234.3285065, 502.2913513, -753.7044678, 780.4907227
4: -365.6630554, 464.3353882, -338.6447449, 428.9564514, -794.6195068, 802.9801025

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099339, upper bound: 9670.6087643
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099339, upper bound: 9670.6087643
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4301.3925781, 5438.3374023, -5371.0097656, 6270.3989258, -10571.7900391, 10809.3476562
1: -512.5007324, 457.1300964, -587.7969971, 547.1542969, -1059.6550293, 1044.9268799
2: -295.7299500, 508.5869751, -353.5109863, 598.4132080, -894.1430664, 862.0977173
3: -242.1795807, 526.8108521, -299.1748657, 613.7947998, -855.9743652, 825.9857178
4: -353.1265259, 447.8102722, -421.9797363, 530.8468628, -883.9732666, 869.7898560

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4190.8535156, 5287.8325195, -7283.0673828, 8061.0117188, -12251.8642578, 12570.8994141
1: -498.0579834, 444.9915466, -752.1517334, 719.0485229, -1217.1064453, 1197.1430664
2: -287.9614868, 494.6303101, -464.5296326, 780.6839600, -1068.6452637, 959.1599121
3: -235.9778748, 512.2885132, -401.9316406, 796.1866455, -1032.1645508, 914.2201538
4: -343.6779480, 435.6154175, -555.1582642, 696.2854004, -1039.9622803, 990.7736816

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4497.0419922, 5680.5131836, -5459.4038086, 6347.7827148, -10844.8242188, 11139.9150391
1: -535.4027710, 477.2313538, -595.0541382, 554.9103394, -1090.3131104, 1072.2854004
2: -308.7634277, 531.1755981, -358.5268555, 606.6144409, -915.3778687, 889.7023926
3: -253.0600281, 550.4812012, -303.8645020, 622.0175781, -875.0776367, 854.3456421
4: -368.3409729, 467.8441162, -427.9151306, 538.2435303, -906.5844116, 895.7592163

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4384.2895508, 5528.8208008, -7381.4199219, 8163.0820312, -12547.3681641, 12910.2402344
1: -520.8402710, 464.9337158, -761.6649170, 728.1735840, -1249.0136719, 1226.5982666
2: -300.8963013, 517.0760498, -470.3925476, 790.8267212, -1091.7229004, 987.4685669
3: -246.7593231, 535.8069458, -407.1517639, 806.3489380, -1053.1082764, 942.9587402
4: -358.7705078, 455.5349731, -562.2100830, 705.1837769, -1063.4443359, 1017.7450562

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.68 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6025183
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097773
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097773
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6086913
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6086913
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6038207
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6050793
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.3331563, upper bound: 9670.3735266
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.3331563, upper bound: 9670.3735266
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6096901, upper bound: 9670.6087643
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6096901, upper bound: 9670.6087643
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6099339, upper bound: 9670.6087643
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6099339, upper bound: 9670.6087643
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6013623, upper bound: 9670.5652639
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -9670.6087240, upper bound: 9670.6087240

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3866.4113770, 4818.6318359, -3930.9780273, 4883.7163086, -8750.1269531, 8749.6083984
1: -453.3428955, 407.7506104, -459.3248901, 413.8519287, -867.1948242, 867.0753784
2: -263.7917175, 452.0695496, -267.7149353, 458.5202942, -722.3118286, 719.7844849
3: -217.2581024, 466.8383484, -220.7507629, 473.3298645, -690.5879517, 687.5891113
4: -314.7858276, 398.7547913, -319.5362854, 404.4896851, -719.2754517, 718.2909546

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.5444510
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5977004
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3996.2277832, 4998.2446289, -3930.9780273, 4883.7163086, -8879.9423828, 8929.2207031
1: -470.4788208, 421.8811035, -459.3248901, 413.8519287, -884.3307495, 881.2059937
2: -272.9423218, 468.3528748, -267.7149353, 458.5202942, -731.4624023, 736.0677490
3: -224.6912842, 484.1605835, -220.7507629, 473.3298645, -698.0211182, 704.9113770
4: -325.4349976, 413.0408325, -319.5362854, 404.4896851, -729.9246826, 732.5770264

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.5519000
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6025183
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3866.4113770, 4818.6318359, -5228.2700195, 6199.5429688, -10065.9541016, 10046.9023438
1: -453.3428955, 407.7506104, -581.7132568, 537.5338745, -990.8767700, 989.4636841
2: -263.7917175, 452.0695496, -347.2168579, 589.5104370, -853.3020020, 799.2863770
3: -217.2581024, 466.8383484, -292.0276184, 605.2127686, -822.4708252, 758.8659058
4: -314.7858276, 398.7547913, -414.8259277, 521.9668579, -836.7526855, 813.5805664

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5627352, upper bound: 9670.5811866
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3996.2277832, 4998.2446289, -5228.2700195, 6199.5429688, -10195.7705078, 10226.5146484
1: -470.4788208, 421.8811035, -581.7132568, 537.5338745, -1008.0125732, 1003.5943604
2: -272.9423218, 468.3528748, -347.2168579, 589.5104370, -862.4525757, 815.5697021
3: -224.6912842, 484.1605835, -292.0276184, 605.2127686, -829.9039917, 776.1882324
4: -325.4349976, 413.0408325, -414.8259277, 521.9668579, -847.4018555, 827.8666382

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5627352, upper bound: 9670.6005063
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5097.9711914, 6050.3696289, -4017.9284668, 4986.1933594, -10084.1640625, 10068.2949219
1: -567.2864990, 524.3539429, -468.5183411, 423.2058411, -990.4922485, 992.8723145
2: -338.9813232, 574.8457031, -274.3129578, 468.0540161, -807.0353394, 849.1586914
3: -284.9885254, 590.1438599, -225.6781158, 483.8479309, -768.8364258, 815.8219604
4: -404.4276428, 509.1234436, -326.6222839, 413.2391663, -817.6668091, 835.7456055

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6002263
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6008968
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7004.2597656, 7815.3867188, -4017.9284668, 4986.1933594, -11990.4531250, 11833.3154297
1: -729.6674805, 695.4097900, -468.5183411, 423.2058411, -1152.8732910, 1163.9277344
2: -449.2025757, 755.1367188, -274.3129578, 468.0540161, -917.2565918, 1029.4497070
3: -387.4227905, 771.0637207, -225.6781158, 483.8479309, -871.2707520, 996.7418213
4: -536.5411987, 673.1435547, -326.6222839, 413.2391663, -949.7803955, 999.7658691

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6002263
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6008968
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5097.9375000, 6050.3325195, -5839.1850586, 6655.8461914, -11753.7802734, 11889.5156250
1: -567.2833252, 524.3505249, -623.1582031, 586.9758301, -1154.2591553, 1147.5084229
2: -338.9790344, 574.8422852, -379.1402893, 639.3779297, -978.3569336, 953.9825439
3: -284.9865417, 590.1403198, -323.3581848, 655.3825684, -940.3690796, 913.4985352
4: -404.4250183, 509.1201782, -452.9455872, 568.6832886, -973.1082764, 962.0657349

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5582877, upper bound: 9670.5995959
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7004.2597656, 7815.3867188, -5839.1850586, 6655.8461914, -13660.1044922, 13654.5712891
1: -729.6674805, 695.4097900, -623.1582031, 586.9758301, -1316.6433105, 1318.5678711
2: -449.2025757, 755.1367188, -379.1402893, 639.3779297, -1088.5805664, 1134.2769775
3: -387.4227905, 771.0637207, -323.3581848, 655.3825684, -1041.8250732, 1094.4218750
4: -536.5411987, 673.1435547, -452.9455872, 568.6832886, -1105.2243652, 1126.0891113

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5582877, upper bound: 9670.5982192
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3866.4113770, 4818.6318359, -4076.0263672, 5080.2885742, -8946.7001953, 8894.6582031
1: -453.3428955, 407.7506104, -478.0450439, 429.4733582, -882.8162842, 885.7954712
2: -263.7917175, 452.0695496, -277.8355713, 476.4255066, -740.2170410, 729.9051514
3: -217.2581024, 466.8383484, -229.0134277, 492.3004761, -709.5585938, 695.8516846
4: -314.7858276, 398.7547913, -331.3485107, 420.2449036, -735.0306396, 730.1032715

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.5444510
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5992305
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3996.2277832, 4998.2446289, -4076.0263672, 5080.2885742, -9076.5166016, 9074.2705078
1: -470.4788208, 421.8811035, -478.0450439, 429.4733582, -899.9521484, 899.9261475
2: -272.9423218, 468.3528748, -277.8355713, 476.4255066, -749.3676758, 746.1884766
3: -224.6912842, 484.1605835, -229.0134277, 492.3004761, -716.9917603, 713.1739502
4: -325.4349976, 413.0408325, -331.3485107, 420.2449036, -745.6798706, 744.3893433

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.6108432
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6108432
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3866.4113770, 4818.6318359, -5416.7402344, 6444.3969727, -10310.8085938, 10235.3720703
1: -453.3428955, 407.7506104, -604.8671875, 557.6233521, -1010.9662476, 1012.6176758
2: -263.7917175, 452.0695496, -360.2341919, 612.0624390, -875.8540649, 812.3037109
3: -217.2581024, 466.8383484, -302.6403809, 628.5065918, -845.7646484, 769.4786987
4: -314.7858276, 398.7547913, -430.1502075, 541.6819458, -856.4677124, 828.9050293

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5627352, upper bound: 9670.5825934
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3996.2277832, 4998.2446289, -5416.7402344, 6444.3969727, -10440.6240234, 10414.9843750
1: -470.4788208, 421.8811035, -604.8671875, 557.6233521, -1028.1021729, 1026.7482910
2: -272.9423218, 468.3528748, -360.2341919, 612.0624390, -885.0046997, 828.5870361
3: -224.6912842, 484.1605835, -302.6403809, 628.5065918, -853.1978149, 786.8009644
4: -325.4349976, 413.0408325, -430.1502075, 541.6819458, -867.1168823, 843.1910400

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5627352, upper bound: 9670.6090792
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097772
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097773
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5177.7998047, 6118.7153320, -4172.5644531, 5190.9453125, -10368.7451172, 10291.2792969
1: -573.7739258, 531.2402954, -487.9042358, 439.7306824, -1013.5046387, 1019.1445312
2: -343.3472290, 582.0405884, -285.0172729, 486.6906433, -830.0378418, 867.0578613
3: -289.1154785, 597.4851074, -234.4779358, 503.6874084, -792.8028564, 831.9630127
4: -409.6969299, 515.6657715, -339.1093445, 429.8288879, -839.5256958, 854.7750244

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7088.5874023, 7904.4028320, -4172.5644531, 5190.9453125, -12279.5322266, 12076.9667969
1: -737.9699707, 703.2954712, -487.9042358, 439.7306824, -1177.7004395, 1191.1997070
2: -454.2579041, 763.8657837, -285.0172729, 486.6906433, -940.9485474, 1048.8830566
3: -391.9350281, 779.8442383, -234.4779358, 503.6874084, -895.6224365, 1014.3221436
4: -542.6865234, 680.7761230, -339.1093445, 429.8288879, -972.5152588, 1019.8441772

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5140.0639648, 6107.1069336, -6002.3598633, 6867.6982422, -12007.7617188, 12109.4658203
1: -572.8653564, 529.0067749, -643.4290771, 604.8945923, -1177.7600098, 1172.4355469
2: -341.9366150, 580.1449585, -390.7414856, 658.8297729, -1000.7663574, 970.8864746
3: -287.1912842, 595.6480103, -332.8655090, 675.3732300, -962.5645142, 928.5135498
4: -407.9547424, 513.8019409, -466.5079041, 585.9307251, -993.8854980, 980.3098145

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6030021, upper bound: 9670.6038193
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6030021, upper bound: 9670.6038193
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5292.0546875, 6270.1567383, -5774.2075195, 6640.9809570, -11933.0341797, 12044.3642578
1: -588.0601807, 543.8732910, -622.4330444, 583.5035400, -1171.5635986, 1166.3063965
2: -351.6752625, 596.1840210, -377.0165710, 636.2006836, -987.8759766, 973.2005615
3: -296.1498108, 611.5601807, -320.6387329, 652.1836548, -948.3334351, 932.1989136
4: -419.2483215, 528.5933228, -449.8818359, 565.6090698, -984.8574219, 978.4751587

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6050793
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6050793
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4091.5583496, 5159.6738281, -3735.9919434, 4717.1445312, -8808.7021484, 8895.6660156
1: -486.5374146, 433.3121033, -444.5668335, 396.2758179, -882.8131104, 877.8789062
2: -279.8354797, 482.7376099, -256.2567139, 440.8562622, -720.6914673, 738.9943237
3: -230.2422791, 499.3995972, -210.5173798, 456.3711853, -686.6134644, 709.9169312
4: -334.3778687, 424.7078857, -305.4596863, 388.4521484, -722.8300171, 730.1676025

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.5591514
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.6087643
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5521.1054688, 6556.4311523, -3735.9919434, 4717.1445312, -10238.2500000, 10292.4208984
1: -616.1984863, 566.9794922, -444.5668335, 396.2758179, -1012.4742432, 1011.5463257
2: -365.8387756, 623.3313599, -256.2567139, 440.8562622, -806.6950684, 879.5880737
3: -307.9964600, 639.7510376, -210.5173798, 456.3711853, -764.3675537, 850.2684326
4: -437.3548279, 550.8035278, -305.4596863, 388.4521484, -825.8068848, 856.2631836

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.5591514
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.6087643
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4233.9365234, 5333.5039062, -4175.6982422, 5179.9926758, -9413.9296875, 9509.2011719
1: -502.8917847, 448.1710205, -487.3818054, 438.7275085, -941.6192627, 935.5527954
2: -289.5798340, 499.3103638, -283.7781677, 486.3761597, -775.9559326, 783.0885010
3: -238.1342010, 516.4805908, -234.3285065, 502.2913513, -740.4255371, 750.8090820
4: -345.9043579, 439.3984680, -338.6447449, 428.9564514, -774.8608398, 778.0432129

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099335, upper bound: 9670.6087643
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099248, upper bound: 9670.6087643
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5668.4384766, 6746.3789062, -4175.6982422, 5179.9926758, -10848.4316406, 10922.0771484
1: -634.0241089, 582.8494263, -487.3818054, 438.7275085, -1072.7515869, 1070.2312012
2: -376.2727661, 641.0275879, -283.7781677, 486.3761597, -862.6488647, 924.8057861
3: -316.2057190, 658.2196045, -234.3285065, 502.2913513, -818.4970703, 892.5480957
4: -449.7722473, 566.3903809, -338.6447449, 428.9564514, -878.7286987, 905.0351562

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099335, upper bound: 9670.6087643
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6099248, upper bound: 9670.6087643
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4134.6699219, 5198.0625000, -5371.0097656, 6270.3989258, -10405.0654297, 10569.0722656
1: -489.3778687, 438.1501465, -587.7969971, 547.1542969, -1036.5322266, 1025.9470215
2: -283.6232910, 486.6148376, -353.5109863, 598.4132080, -882.0363770, 840.1256104
3: -232.6967468, 503.7722168, -299.1748657, 613.7947998, -846.4915771, 802.9470825
4: -338.3375854, 428.6560364, -421.9797363, 530.8468628, -869.1843262, 850.6357422

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6003909, upper bound: 9670.5652639
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022500, upper bound: 9670.5652639
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5956.2050781, 6841.6093750, -5371.0097656, 6270.3989258, -12226.6035156, 12212.6191406
1: -641.4949951, 601.0505981, -587.7969971, 547.1542969, -1188.6492920, 1188.8475342
2: -387.9178162, 655.8599243, -353.5109863, 598.4132080, -986.3309326, 1009.3708496
3: -330.2873840, 672.1271973, -299.1748657, 613.7947998, -944.0821533, 971.3020020
4: -463.7815857, 582.3920898, -421.9797363, 530.8468628, -994.6284180, 1004.3718262

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6003909, upper bound: 9670.5652639
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6022500, upper bound: 9670.5652639
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4134.6572266, 5198.0498047, -7283.0673828, 8061.0117188, -12195.6689453, 12481.1171875
1: -489.3767090, 438.1490173, -752.1517334, 719.0485229, -1208.4251709, 1190.3005371
2: -283.6225281, 486.6136475, -464.5296326, 780.6839600, -1064.3063965, 951.1433105
3: -232.6960449, 503.7710571, -401.9316406, 796.1866455, -1028.8826904, 905.7026978
4: -338.3367310, 428.6550293, -555.1582642, 696.2854004, -1034.3789062, 983.8132935

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5956.2050781, 6841.6093750, -7283.0673828, 8061.0117188, -14017.2167969, 14124.6767578
1: -641.4949951, 601.0505981, -752.1517334, 719.0485229, -1360.5433350, 1353.2022705
2: -387.9178162, 655.8599243, -464.5296326, 780.6839600, -1168.6015625, 1120.3895264
3: -330.2873840, 672.1271973, -401.9316406, 796.1866455, -1126.4739990, 1073.3999023
4: -463.7815857, 582.3920898, -555.1582642, 696.2854004, -1159.8664551, 1137.5502930

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4322.7358398, 5431.8496094, -5459.4038086, 6347.7827148, -10670.5185547, 10891.2539062
1: -511.5397949, 457.4799194, -595.0541382, 554.9103394, -1066.4501953, 1052.5336914
2: -296.1861572, 508.3769531, -358.5268555, 606.6144409, -902.8005981, 866.9036865
3: -243.1773834, 526.6154785, -303.8645020, 622.0175781, -865.1949463, 830.4799194
4: -352.9688721, 447.9653931, -427.9151306, 538.2435303, -891.2122803, 875.8804932

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6039272, upper bound: 9670.6035871
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6035871
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6248.2309570, 7242.2172852, -5459.4038086, 6347.7827148, -12596.0136719, 12701.6210938
1: -680.3137207, 633.2691650, -595.0541382, 554.9103394, -1235.2238770, 1228.3231201
2: -408.5906677, 692.8860474, -358.5268555, 606.6144409, -1015.2050781, 1051.4127197
3: -346.7114258, 710.7530518, -303.8645020, 622.0175781, -968.7289429, 1014.6174927
4: -488.2898254, 614.3981934, -427.9151306, 538.2435303, -1026.5333252, 1042.3133545

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6039272, upper bound: 9670.6050476
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6050476
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4322.7187500, 5431.8305664, -7381.4199219, 8163.0820312, -12485.7988281, 12813.2500000
1: -511.5381775, 457.4781799, -761.6649170, 728.1735840, -1239.7115479, 1219.1427002
2: -296.1849060, 508.3751831, -470.3925476, 790.8267212, -1087.0115967, 978.7675781
3: -243.1763611, 526.6136475, -407.1517639, 806.3489380, -1049.5252686, 933.7653198
4: -352.9675293, 447.9637451, -562.2100830, 705.1837769, -1057.4110107, 1010.1737671

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6030619, upper bound: 9670.6035871
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6035871
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6248.2309570, 7242.2172852, -7381.4199219, 8163.0820312, -14411.3105469, 14623.6367188
1: -680.3137207, 633.2691650, -761.6649170, 728.1735840, -1408.4870605, 1394.9338379
2: -408.5906677, 692.8860474, -470.3925476, 790.8267212, -1199.4173584, 1163.2784424
3: -346.7114258, 710.7530518, -407.1517639, 806.3489380, -1152.8187256, 1117.6053467
4: -488.2898254, 614.3981934, -562.2100830, 705.1837769, -1192.9918213, 1176.6081543

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6030619, upper bound: 9670.6050476
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6050476
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.29 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.5444510
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5977004
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.5519000
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6025183
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5871596
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6026556
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6002263
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6008968
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6002263
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6008968
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6020540
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6009239
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.5444510
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5992305
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4889525, upper bound: 9670.6108432
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6108432
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5893108
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097772
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6097773
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6087643, upper bound: 9670.6086913
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6030021, upper bound: 9670.6038193
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6030021, upper bound: 9670.6038193
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6050793
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6050476, upper bound: 9670.6050793
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.5591514
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.6087643
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.5591514
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.4562563, upper bound: 9670.6087643
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6099335, upper bound: 9670.6087643
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6099248, upper bound: 9670.6087643
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6099335, upper bound: 9670.6087643
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6099248, upper bound: 9670.6087643
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6003909, upper bound: 9670.5652639
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6022500, upper bound: 9670.5652639
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6003909, upper bound: 9670.5652639
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6022500, upper bound: 9670.5652639
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6009239, upper bound: 9670.5652639
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6039272, upper bound: 9670.6035871
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6035871
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6039272, upper bound: 9670.6050476
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6050476
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6030619, upper bound: 9670.6035871
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6035871
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6030619, upper bound: 9670.6050476
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -9670.6035870, upper bound: 9670.6050476

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3773.0229492, 4712.5302734, -3783.7238770, 4778.9145508, -8551.9365234, 8496.2539062
1: -443.5846558, 398.3393250, -450.6090088, 401.4846191, -845.0692139, 848.9481201
2: -257.6374207, 441.7334595, -259.4738159, 447.6551819, -705.2926025, 701.2072754
3: -212.0926971, 456.5293274, -212.9206848, 462.5089417, -674.6015625, 669.4499512
4: -307.4858398, 389.4045105, -309.9125671, 394.0892334, -701.5750732, 699.3170776

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4769286, upper bound: 9670.4264122
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4769286, upper bound: 9670.5346245
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3816.3542480, 4765.7846680, -3808.0195312, 4753.8452148, -8570.1982422, 8573.8046875
1: -448.5119019, 402.8576050, -447.4248962, 401.8333130, -850.3452148, 850.2824707
2: -260.6160278, 446.9176025, -259.9201355, 445.8628540, -706.4788818, 706.8377075
3: -214.4846039, 461.6104431, -213.9620209, 460.4751282, -674.9597168, 675.5724487
4: -310.9685669, 394.1405334, -310.1756897, 393.1561890, -704.1245728, 704.3161621

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5110439, upper bound: 9670.4264122
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5677917, upper bound: 9670.5957511
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3898.1938477, 4885.7324219, -3783.7238770, 4778.9145508, -8677.1054688, 8669.4560547
1: -460.1472778, 411.9347839, -450.6090088, 401.4846191, -861.6317749, 862.5436401
2: -266.4603271, 457.4344177, -259.4738159, 447.6551819, -714.1154785, 716.9082031
3: -219.2008514, 473.2201233, -212.9206848, 462.5089417, -681.7097778, 686.1408081
4: -317.7425842, 403.2054443, -309.9125671, 394.0892334, -711.8317871, 713.1179199

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4739593, upper bound: 9670.5363386
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.4739593, upper bound: 9670.4998165
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3949.7700195, 4948.7011719, -3808.0195312, 4753.8452148, -8703.6152344, 8756.7187500
1: -465.9525146, 417.3176270, -447.4248962, 401.8333130, -867.7858276, 864.7425537
2: -269.9739990, 463.5402832, -259.9201355, 445.8628540, -715.8368530, 723.4604492
3: -222.1060181, 479.2858887, -213.9620209, 460.4751282, -682.5811768, 693.2477417
4: -321.8709106, 408.7157898, -310.1756897, 393.1561890, -715.0270386, 718.8914185

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5415766, upper bound: 9670.5673847
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5415766, upper bound: 9670.5945201
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3866.4113770, 4818.6318359, -5155.2836914, 6130.2290039, -9996.6406250, 9973.9160156
1: -453.3428955, 407.7506104, -575.3083496, 530.8390503, -984.1818848, 983.0588379
2: -263.7917175, 452.0695496, -342.9400024, 582.5305786, -846.3221436, 795.0095215
3: -217.2581024, 466.8383484, -288.0801086, 598.1308594, -815.3889160, 754.9183960
4: -314.7858276, 398.7547913, -409.6829834, 515.5894165, -830.3751831, 808.4377441

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5683695, upper bound: 9670.5840258
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5683695, upper bound: 9670.5622946
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3866.4113770, 4818.6318359, -5346.9794922, 6403.6201172, -10270.0302734, 10165.6113281
1: -453.3428955, 407.7506104, -601.7843018, 552.1317139, -1005.4746094, 1009.5348511
2: -263.7917175, 452.0695496, -356.3210144, 607.7003784, -871.4919434, 808.3905640
3: -217.2581024, 466.8383484, -298.8876953, 624.1907959, -841.4489136, 765.7260742
4: -314.7858276, 398.7547913, -426.3570557, 536.5319214, -851.3176270, 825.1118164

Time for backsubstitution: 3.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5683695, upper bound: 9670.5840258
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5683695, upper bound: 9670.5622946
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3996.2277832, 4998.2446289, -5155.2836914, 6130.2290039, -10126.4560547, 10153.5283203
1: -470.4788208, 421.8811035, -575.3083496, 530.8390503, -1001.3178101, 997.1894531
2: -272.9423218, 468.3528748, -342.9400024, 582.5305786, -855.4727173, 811.2926636
3: -224.6912842, 484.1605835, -288.0801086, 598.1308594, -822.8220825, 772.2407227
4: -325.4349976, 413.0408325, -409.6829834, 515.5894165, -841.0244141, 822.7238159

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5683695, upper bound: 9670.6023412
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5679101, upper bound: 9670.6024536
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3996.2277832, 4998.2446289, -5346.9794922, 6403.6201172, -10399.8466797, 10345.2246094
1: -470.4788208, 421.8811035, -601.7843018, 552.1317139, -1022.6105347, 1023.6654053
2: -272.9423218, 468.3528748, -356.3210144, 607.7003784, -880.6425781, 824.6737671
3: -224.6912842, 484.1605835, -298.8876953, 624.1907959, -848.8820801, 783.0482788
4: -325.4349976, 413.0408325, -426.3570557, 536.5319214, -861.9668579, 839.3978882

Time for backsubstitution: 3.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5683695, upper bound: 9670.6023412
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5679101, upper bound: 9670.6024536
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5097.9711914, 6050.3696289, -3754.0463867, 4625.0981445, -9723.0683594, 9804.4121094
1: -567.2864990, 524.3539429, -434.3667603, 393.4705505, -960.7570801, 958.7207031
2: -338.9813232, 574.8457031, -254.7405548, 435.1082458, -774.0895996, 829.5860596
3: -284.9885254, 590.1438599, -210.6668396, 448.8341675, -733.8226318, 800.8106689
4: -404.4276428, 509.1234436, -303.6756897, 384.0916138, -788.5192871, 812.7991333

Time for backsubstitution: 3.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5936549
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6014958
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5097.9711914, 6050.3696289, -5046.4765625, 5924.4580078, -11022.4296875, 11096.8447266
1: -567.2864990, 524.3539429, -555.3024902, 516.1605225, -1083.4466553, 1079.6561279
2: -338.9813232, 574.8457031, -333.4494934, 564.7029419, -903.6842651, 908.2951660
3: -284.9885254, 590.1438599, -281.5552368, 579.6424561, -864.6309814, 871.6989746
4: -404.4276428, 509.1234436, -398.2368469, 500.7270508, -905.1546631, 907.3601685

Time for backsubstitution: 3.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.5936550
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5699626, upper bound: 9670.6014958
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7004.2597656, 7815.3867188, -3754.0463867, 4625.0981445, -11629.3564453, 11569.4326172
1: -729.6674805, 695.4097900, -434.3667603, 393.4705505, -1123.1379395, 1129.7763672
2: -449.2025757, 755.1367188, -254.7405548, 435.1082458, -884.3107910, 1009.8771362
3: -387.4227905, 771.0637207, -210.6668396, 448.8341675, -836.2569580, 981.7305908
4: -536.5411987, 673.1435547, -303.6756897, 384.0916138, -920.6328125, 976.8192139

Time for backsubstitution: 3.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5657787, upper bound: 9670.5968067
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5657787, upper bound: 9670.5988362
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7004.2597656, 7815.3867188, -5046.4765625, 5924.4580078, -12928.7177734, 12861.8632812
1: -729.6674805, 695.4097900, -555.3024902, 516.1605225, -1245.8280029, 1250.7120361
2: -449.2025757, 755.1367188, -333.4494934, 564.7029419, -1013.9055176, 1088.5860596
3: -387.4227905, 771.0637207, -281.5552368, 579.6424561, -967.0652466, 1052.6188965
4: -536.5411987, 673.1435547, -398.2368469, 500.7270508, -1037.2681885, 1071.3803711

Time for backsubstitution: 3.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5657787, upper bound: 9670.5988484
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5657787, upper bound: 9670.6003965
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5097.9375000, 6050.3325195, -5776.8037109, 6594.7797852, -11692.7138672, 11827.1337891
1: -567.2833252, 524.3505249, -617.5706787, 581.2160034, -1148.4992676, 1141.9210205
2: -338.9790344, 574.8422852, -375.4524841, 633.2161865, -972.1951904, 950.2947998
3: -284.9865417, 590.1403198, -319.9759521, 649.2439575, -934.2304688, 910.1162720
4: -404.4250183, 509.1201782, -448.4918823, 563.1358643, -967.5609131, 957.6118164

Time for backsubstitution: 3.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.5859309
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6026354
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5097.9375000, 6050.3325195, -5849.3227539, 6724.8144531, -11822.7490234, 11899.6533203
1: -567.2833252, 524.3505249, -630.5406494, 590.3935547, -1157.6767578, 1154.8908691
2: -338.9790344, 574.8422852, -381.0140381, 644.4453735, -983.4243774, 955.8562622
3: -284.9865417, 590.1403198, -324.2935791, 660.7409668, -945.7274780, 914.4338989
4: -404.4250183, 509.1201782, -455.5789795, 572.1582031, -976.5832520, 964.6989746

Time for backsubstitution: 3.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.5859309
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6026354
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7004.2597656, 7815.3867188, -5776.8037109, 6594.7797852, -13599.0380859, 13592.1894531
1: -729.6674805, 695.4097900, -617.5706787, 581.2160034, -1310.8835449, 1312.9804688
2: -449.2025757, 755.1367188, -375.4524841, 633.2161865, -1082.4187012, 1130.5892334
3: -387.4227905, 771.0637207, -319.9759521, 649.2439575, -1035.7524414, 1091.0396729
4: -536.5411987, 673.1435547, -448.4918823, 563.1358643, -1099.6770020, 1121.6354980

Time for backsubstitution: 3.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.5988484
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6004278
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7004.2597656, 7815.3867188, -5849.3227539, 6724.8144531, -13729.0732422, 13664.7080078
1: -729.6674805, 695.4097900, -630.5406494, 590.3935547, -1320.0610352, 1325.9503174
2: -449.2025757, 755.1367188, -381.0140381, 644.4453735, -1093.6479492, 1136.1507568
3: -387.4227905, 771.0637207, -324.2935791, 660.7409668, -1047.5966797, 1095.3572998
4: -536.5411987, 673.1435547, -455.5789795, 572.1582031, -1108.6993408, 1128.7225342

Time for backsubstitution: 3.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.5988484
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9670.5652639, upper bound: 9670.6004278
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3773.0229492, 4712.5302734, -3911.7126465, 4953.7563477, -8726.7783203, 8624.2431641
1: -443.5846558, 398.3393250, -467.4323120, 415.2018738, -858.7864990, 865.7716064
2: -257.6374207, 441.7334595, -268.4620056, 463.5293579, -721.1666260, 710.1954346
3: -212.0926971, 456.5293274, -220.1319885, 479.4873657, -691.5798950, 676.6612549
4: -307.4858398, 389.4045105, -320.2908325, 408.0913391, -715.5771484, 709.6953125

Time for backsubstitution: 3.34 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.64 + 418.26 = 422.89 seconds
