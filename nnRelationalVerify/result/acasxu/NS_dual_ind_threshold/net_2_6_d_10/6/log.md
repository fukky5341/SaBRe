## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 3295.06222004829


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6975.3520508, 12629.0117188, -6975.3520508, 12629.0117188, -19604.3632812, 19604.3632812)
1: (-1367.2161865, 1793.9888916, -1367.2161865, 1793.9888916, -3161.2050781, 3161.2050781)
2: (-1009.3267822, 2079.8847656, -1009.3267822, 2079.8847656, -3089.2114258, 3089.2114258)
3: (-1079.8317871, 2775.1779785, -1079.8317871, 2775.1779785, -3855.0090332, 3855.0090332)
4: (-860.8695068, 2594.6042480, -860.8695068, 2594.6042480, -3455.4736328, 3455.4736328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.43 + 2.19 = 4.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -3295.0951710, upper bound: 3295.0951709

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0936398
time: 0.80 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0936995, upper bound: 3295.0936996
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.86 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0936398
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 3, lower bound: -3295.0936995, upper bound: 3295.0936996

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6712.3061523, 12160.8310547, -6805.2792969, 12325.2939453, -19037.5996094, 18966.1093750
1: -1315.4766846, 1726.8565674, -1333.6514893, 1750.5275879, -3066.0041504, 3060.5080566
2: -971.5372925, 2002.2072754, -984.8374634, 2029.6968994, -3001.2341309, 2987.0446777
3: -1038.6553955, 2672.6721191, -1053.1970215, 2708.7565918, -3747.4116211, 3725.8691406
4: -828.3814697, 2498.4587402, -839.8204956, 2532.4892578, -3360.8706055, 3338.2788086

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0892150, upper bound: 3295.0911281
time: 0.85 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0892079, upper bound: 3295.0888442
time: 1.04 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7417.5991211, 13374.9853516, -6870.3359375, 12434.9541016, -19852.5527344, 20245.3203125
1: -1441.8498535, 1896.5756836, -1346.1413574, 1766.5756836, -3208.4252930, 3242.7170410
2: -1068.5217285, 2199.3256836, -993.8072510, 2047.4963379, -3116.0178223, 3193.1328125
3: -1144.7545166, 2944.7966309, -1063.4912109, 2732.8493652, -3877.6037598, 4008.2878418
4: -911.7056885, 2743.9956055, -847.5876465, 2554.4721680, -3466.1774902, 3591.5825195

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0936984, upper bound: 3295.0936879
time: 0.90 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0936880, upper bound: 3295.0936874
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.16 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -3295.0892150, upper bound: 3295.0911281
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -3295.0892079, upper bound: 3295.0888442
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -3295.0936984, upper bound: 3295.0936879
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 3, lower bound: -3295.0936880, upper bound: 3295.0936874

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6626.0424805, 12004.9394531, -6673.7392578, 12087.1582031, -18713.2011719, 18678.6738281
1: -1298.9051514, 1704.7891846, -1308.3209229, 1716.9473877, -3015.8525391, 3013.1101074
2: -959.1024780, 1976.4174805, -965.8980103, 1990.2570801, -2949.3596191, 2942.3149414
3: -1025.4323730, 2637.9064941, -1032.9760742, 2655.6269531, -3681.0590820, 3670.8820801
4: -817.7653809, 2466.3928223, -823.6382446, 2483.4641113, -3301.2292480, 3290.0310059

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0892129, upper bound: 3295.0908524
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0892129, upper bound: 3295.0911280
time: 0.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6597.6635742, 11949.9257812, -6992.8032227, 12641.4941406, -19239.1582031, 18942.7285156
1: -1292.5113525, 1696.6862793, -1367.3896484, 1794.9570312, -3087.4675293, 3064.0751953
2: -954.6129761, 1968.1925049, -1010.4984741, 2083.0209961, -3037.6340332, 2978.6909180
3: -1020.8031006, 2627.2734375, -1083.9390869, 2779.8857422, -3800.6887207, 3711.2124023
4: -813.9367065, 2456.2927246, -862.7089844, 2598.5085449, -3412.4448242, 3319.0014648

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0851031, upper bound: 3295.0810994
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0822384, upper bound: 3295.0810989
time: 0.89 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7362.3911133, 13274.1523438, -6779.9443359, 12267.4667969, -19629.8574219, 20054.0976562
1: -1430.3824463, 1882.3703613, -1327.7091064, 1743.1636963, -3173.5461426, 3210.0795898
2: -1060.5211182, 2183.4570312, -980.5397339, 2021.2174072, -3081.7385254, 3163.9968262
3: -1136.2672119, 2922.8466797, -1049.3264160, 2696.7883301, -3833.0556641, 3972.1728516
4: -904.8989258, 2724.2792969, -836.3005981, 2521.7692871, -3426.6682129, 3560.5795898

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0904293, upper bound: 3295.0888156
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0861308, upper bound: 3295.0881494
time: 0.87 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7307.1259766, 13170.1660156, -7452.6826172, 13467.2304688, -20774.3535156, 20622.8417969
1: -1419.5098877, 1867.0261230, -1450.6384277, 1903.4581299, -3322.9680176, 3317.6640625
2: -1052.0281982, 2165.8571777, -1073.5971680, 2206.0739746, -3258.1020508, 3239.4543457
3: -1127.2956543, 2900.1752930, -1148.3526611, 2963.4309082, -4090.7260742, 4048.5278320
4: -897.7011719, 2702.2209473, -915.7770996, 2752.2729492, -3649.9738770, 3617.9980469

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861244
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0861247, upper bound: 3295.0861248
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0892129, upper bound: 3295.0908524
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0892129, upper bound: 3295.0911280
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0851031, upper bound: 3295.0810994
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0822384, upper bound: 3295.0810989
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0904293, upper bound: 3295.0888156
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0861308, upper bound: 3295.0881494
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861244
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -3295.0861247, upper bound: 3295.0861248

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6468.1630859, 11714.7568359, -6568.5961914, 11893.8886719, -18362.0507812, 18283.3535156
1: -1266.8873291, 1664.3016357, -1287.0184326, 1689.9543457, -2956.8417969, 2951.3195801
2: -936.4295654, 1929.6264648, -950.7855835, 1959.0731201, -2895.5026855, 2880.4121094
3: -1001.1571655, 2574.4772949, -1016.8236084, 2613.3901367, -3614.5466309, 3591.3005371
4: -798.3775635, 2408.3549805, -810.7332153, 2444.7875977, -3243.1650391, 3219.0881348

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827807, upper bound: 3295.0861673
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840656
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6794.8178711, 12307.4228516, -6647.4765625, 12039.9228516, -18834.7402344, 18954.8925781
1: -1331.2971191, 1747.7860107, -1303.1545410, 1710.2241211, -3041.5212402, 3050.9404297
2: -983.3768311, 2027.1500244, -962.1250610, 1982.4700928, -2965.8466797, 2989.2751465
3: -1051.6169434, 2704.8623047, -1028.9691162, 2645.3137207, -3696.9306641, 3733.8315430
4: -838.5363770, 2529.8850098, -820.4211426, 2473.8293457, -3312.3657227, 3350.3056641

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826963, upper bound: 3295.0861201
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0843894
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6459.3295898, 11704.5146484, -6902.3256836, 12479.8525391, -18939.1757812, 18606.8378906
1: -1266.1671143, 1661.9997559, -1349.9993896, 1772.0725098, -3038.2395020, 3011.9990234
2: -934.9556274, 1927.3352051, -997.5387573, 2056.1169434, -2991.0725098, 2924.8740234
3: -999.5513306, 2572.9587402, -1069.9763184, 2744.2336426, -3743.7849121, 3642.9350586
4: -797.0623169, 2405.4433594, -851.5904541, 2565.0302734, -3362.0925293, 3257.0336914

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824119, upper bound: 3295.0810990
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824096, upper bound: 3295.0808728
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6874.9594727, 12494.2451172, -6868.5849609, 12425.3955078, -19300.3554688, 19362.8281250
1: -1355.6345215, 1770.4960938, -1344.8115234, 1764.0806885, -3119.7148438, 3115.3073730
2: -996.5516968, 2053.8330078, -993.2092285, 2046.6015625, -3043.1528320, 3047.0419922
3: -1065.3990479, 2744.7277832, -1065.4476318, 2731.7109375, -3797.1096191, 3810.1752930
4: -849.8358765, 2562.6647949, -847.9285278, 2553.0585938, -3402.8940430, 3410.5932617

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821161, upper bound: 3295.0810989
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821115, upper bound: 3295.0808738
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7229.0336914, 13030.5019531, -6694.1376953, 12113.2255859, -19342.2597656, 19724.6386719
1: -1404.6536865, 1848.2193604, -1311.2833252, 1721.3417969, -3125.9956055, 3159.5026855
2: -1041.2602539, 2143.6894531, -968.2534180, 1995.6689453, -3036.9289551, 3111.9428711
3: -1115.7180176, 2869.2927246, -1036.2044678, 2662.3515625, -3778.0695801, 3905.4970703
4: -888.4776001, 2674.8562012, -825.8063965, 2489.9987793, -3378.4763184, 3500.6625977

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0838737, upper bound: 3295.0845757
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0837044, upper bound: 3295.0810719
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7511.0664062, 13524.7470703, -6664.0717773, 12054.9267578, -19565.9902344, 20188.8164062
1: -1456.7464600, 1917.3220215, -1304.6459961, 1712.7353516, -3169.4816895, 3221.9680176
2: -1080.6365967, 2224.8679199, -963.4819946, 1987.0319824, -3067.6684570, 3188.3496094
3: -1160.0876465, 2978.7316895, -1031.2187500, 2651.1044922, -3811.1914062, 4009.9504395
4: -922.7381592, 2775.6508789, -821.7160034, 2479.3884277, -3402.1262207, 3597.3666992

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778949, upper bound: 3295.0823033
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0809542
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7176.3789062, 12931.0615234, -7361.1127930, 13301.0507812, -20477.4296875, 20292.1738281
1: -1394.2486572, 1833.5935059, -1432.9688721, 1879.9960938, -3274.2446289, 3266.5622559
2: -1033.1676025, 2126.8737793, -1060.3638916, 2178.8525391, -3212.0195312, 3187.2377930
3: -1107.1300049, 2847.6184082, -1134.2098389, 2926.8544922, -4033.9843750, 3981.8281250
4: -881.6008911, 2653.7761230, -904.4840088, 2718.4240723, -3600.0246582, 3558.2602539

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0893978, upper bound: 3295.0851637
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861245
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7467.1127930, 13443.2792969, -7359.2143555, 13293.8369141, -20760.9492188, 20802.4941406
1: -1448.4146729, 1905.2272949, -1431.7967529, 1878.8507080, -3327.2653809, 3337.0239258
2: -1073.9365234, 2210.8190918, -1059.7553711, 2178.2546387, -3252.1911621, 3270.5739746
3: -1153.0018311, 2960.7766113, -1133.7458496, 2925.8674316, -4078.8691406, 4094.5222168
4: -917.0571899, 2757.9882812, -903.9547119, 2717.7729492, -3634.8300781, 3661.9423828

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0856154, upper bound: 3295.0851637
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0861248, upper bound: 3295.0861247
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.93 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0827807, upper bound: 3295.0861673
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840656
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0826963, upper bound: 3295.0861201
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0843894
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0824119, upper bound: 3295.0810990
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0824096, upper bound: 3295.0808728
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0821161, upper bound: 3295.0810989
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0821115, upper bound: 3295.0808738
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0838737, upper bound: 3295.0845757
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0837044, upper bound: 3295.0810719
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0778949, upper bound: 3295.0823033
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0809542
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0893978, upper bound: 3295.0851637
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861245
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0856154, upper bound: 3295.0851637
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.93
Output dim: 3, lower bound: -3295.0861248, upper bound: 3295.0861247

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6377.4858398, 11553.1943359, -6422.7880859, 11634.4980469, -18011.9843750, 17975.9804688
1: -1249.5606689, 1641.4525146, -1259.1794434, 1653.3577881, -2902.9179688, 2900.6318359
2: -923.5007324, 1902.8161621, -930.0544434, 1916.0319824, -2839.5327148, 2832.8706055
3: -987.2176514, 2538.8574219, -994.4791870, 2556.1186523, -3543.3359375, 3533.3364258
4: -787.2860107, 2374.9990234, -792.9395752, 2391.2021484, -3178.4880371, 3167.9384766

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840042
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827318, upper bound: 3295.0839933
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6348.9282227, 11509.8251953, -6839.5078125, 12431.0478516, -18779.9726562, 18349.3320312
1: -1245.6127930, 1635.0061035, -1349.3188477, 1762.5373535, -3008.1501465, 2984.3242188
2: -919.9774780, 1895.0052490, -992.0584106, 2043.2915039, -2963.2690430, 2887.0637207
3: -983.4656982, 2528.4697266, -1060.7716064, 2728.9321289, -3712.3977051, 3589.2412109
4: -784.2721558, 2365.1203613, -846.0676880, 2549.3127441, -3333.5844727, 3211.1879883

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840041
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827272, upper bound: 3295.0839933
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6708.4790039, 12153.7275391, -6507.1235352, 11790.6728516, -18499.1523438, 18660.8515625
1: -1314.8341064, 1726.0618896, -1276.4239502, 1675.0635986, -2989.8977051, 3002.4858398
2: -971.0791626, 2001.6225586, -942.1983643, 1941.0848389, -2912.1640625, 2943.8205566
3: -1038.3813477, 2670.9802246, -1007.4851685, 2590.2260742, -3628.6071777, 3678.4648438
4: -827.9885864, 2498.1140137, -803.3229370, 2422.2971191, -3250.2856445, 3301.4367676

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840574
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842716
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6669.0136719, 12086.9980469, -6903.9199219, 12548.3369141, -19217.3437500, 18990.9179688
1: -1308.2225342, 1716.4027100, -1362.3034668, 1778.6750488, -3086.8974609, 3078.7055664
2: -965.7342529, 1990.2697754, -1001.1257935, 2061.9790039, -3027.7133789, 2991.3955078
3: -1032.5878906, 2655.6708984, -1070.4990234, 2754.7495117, -3787.3374023, 3726.1699219
4: -823.4212646, 2483.8347168, -853.8308716, 2572.5466309, -3395.9672852, 3337.6650391

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840520
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826765, upper bound: 3295.0842717
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6445.0034180, 11678.9873047, -6839.4472656, 12367.1503906, -18812.1542969, 18518.4316406
1: -1263.4851074, 1658.4288330, -1338.1462402, 1756.2910156, -3019.7761230, 2996.5749512
2: -932.9394531, 1922.9876709, -988.6318970, 2036.9471436, -2969.8867188, 2911.6193848
3: -997.4038086, 2567.2492676, -1060.5512695, 2719.0151367, -3716.4189453, 3627.8005371
4: -795.3388672, 2399.9890137, -844.0166016, 2540.9865723, -3336.3242188, 3244.0056152

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808949
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6445.9858398, 11680.1884766, -7602.1166992, 13812.0537109, -20258.0390625, 19282.3046875
1: -1263.5318604, 1658.5321045, -1496.0201416, 1956.7673340, -3220.2988281, 3154.5522461
2: -933.0110474, 1923.3720703, -1101.3509521, 2265.7902832, -3198.8012695, 3024.7226562
3: -997.4620972, 2567.6564941, -1182.4693604, 3032.7460938, -4030.2077637, 3750.1259766
4: -795.4060669, 2400.4826660, -940.7669067, 2826.9897461, -3622.3947754, 3341.2495117

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808400
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808403
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6862.1650391, 12471.3789062, -6802.5917969, 12306.4501953, -19168.6152344, 19273.9707031
1: -1353.2229004, 1767.2851562, -1332.2502441, 1747.3939209, -3100.6166992, 3099.5354004
2: -994.7472534, 2049.9160156, -983.7994385, 2026.4155273, -3021.1625977, 3033.7153320
3: -1063.4815674, 2739.5986328, -1055.4935303, 2705.1616211, -3768.6428223, 3795.0920410
4: -848.2939453, 2557.7678223, -839.9328613, 2527.7656250, -3376.0590820, 3397.7006836

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820927, upper bound: 3295.0808946
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820347, upper bound: 3295.0808945
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6848.7329102, 12447.2285156, -7523.0429688, 13674.3945312, -20523.1269531, 19970.2714844
1: -1350.5621338, 1763.8406982, -1481.7519531, 1936.7958984, -3287.3576660, 3245.5922852
2: -992.7973022, 2046.0115967, -1090.2564697, 2242.3342285, -3235.1315918, 3136.2680664
3: -1061.3306885, 2734.4084473, -1170.6217041, 3002.1677246, -4063.4985352, 3905.0300293
4: -846.6187134, 2552.8972168, -931.3133545, 2797.7387695, -3644.3574219, 3484.2104492

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821048, upper bound: 3295.0808396
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820339, upper bound: 3295.0808390
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7141.6591797, 12875.3916016, -6551.5952148, 11859.6337891, -19001.2851562, 19426.9863281
1: -1387.9427490, 1826.2946777, -1284.0604248, 1685.5675049, -3073.5100098, 3110.3547363
2: -1028.8382568, 2117.8088379, -947.9646606, 1953.5784912, -2982.4167480, 3065.7731934
3: -1102.2780762, 2834.9621582, -1014.3499756, 2606.2670898, -3708.5451660, 3849.3117676
4: -877.8143921, 2642.6791992, -808.3994751, 2437.5905762, -3315.4050293, 3451.0781250

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0838737, upper bound: 3295.0845756
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0838737, upper bound: 3295.0845759
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7097.2817383, 12803.7783203, -6946.7788086, 12616.2197266, -19713.5019531, 19750.5507812
1: -1380.9555664, 1815.8115234, -1369.7463379, 1788.9422607, -3169.8979492, 3185.5571289
2: -1023.0446167, 2104.8217773, -1006.7315063, 2074.3256836, -3097.3703613, 3111.5532227
3: -1096.0539551, 2818.4638672, -1077.2611084, 2770.5595703, -3866.6135254, 3895.7250977
4: -872.8482666, 2626.4111328, -858.7850952, 2587.6474609, -3460.4956055, 3485.1960449

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0837044, upper bound: 3295.0810718
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0837044, upper bound: 3295.0810717
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7420.3452148, 13363.2128906, -6522.5830078, 11803.3476562, -19223.6875000, 19885.7871094
1: -1439.2917480, 1894.4439697, -1277.6307373, 1677.2480469, -3116.5390625, 3172.0747070
2: -1067.6844482, 2197.8811035, -943.3400269, 1945.2565918, -3012.9406738, 3141.2209473
3: -1146.0769043, 2943.0231934, -1009.5220337, 2595.4125977, -3741.4895020, 3952.5444336
4: -911.6168213, 2742.1289062, -804.4385376, 2427.3720703, -3338.9887695, 3546.5668945

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778949, upper bound: 3295.0823034
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778949, upper bound: 3295.0823032
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7382.8593750, 13302.9785156, -6924.0058594, 12570.3037109, -19953.1640625, 20226.9804688
1: -1433.5090332, 1885.5659180, -1364.5776367, 1782.1503906, -3215.6591797, 3250.1435547
2: -1062.7940674, 2186.8276367, -1002.9802246, 2067.7050781, -3130.4987793, 3189.8078613
3: -1140.8452148, 2929.0793457, -1073.3890381, 2762.1647949, -3903.0100098, 4002.4682617
4: -907.4390259, 2728.2810059, -855.5761108, 2579.5925293, -3487.0307617, 3583.8564453

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0809542
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0809542
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7067.2509766, 12733.9667969, -7212.6440430, 13037.4267578, -20104.6738281, 19946.6113281
1: -1372.2495117, 1805.6201172, -1404.3387451, 1841.9594727, -3214.2089844, 3209.9589844
2: -1017.5650024, 2094.9897461, -1039.2562256, 2135.4116211, -3152.9763184, 3134.2460938
3: -1090.4277344, 2804.3815918, -1111.5086670, 2868.5427246, -3958.9704590, 3915.8901367
4: -868.2561646, 2614.2297363, -886.4045410, 2664.5659180, -3532.8217773, 3500.6342773

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0893820, upper bound: 3295.0851635
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0893820, upper bound: 3295.0851635
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7019.5708008, 12639.8125000, -7369.4453125, 13294.3896484, -20313.9589844, 20009.2578125
1: -1363.0455322, 1791.0356445, -1431.2568359, 1875.9439697, -3238.9895020, 3222.2924805
2: -1009.7023315, 2079.8759766, -1059.5142822, 2178.3088379, -3188.0112305, 3139.3901367
3: -1082.6950684, 2784.2521973, -1134.8846436, 2928.0871582, -4010.1901855, 3919.1367188
4: -861.7362671, 2595.1354980, -903.9960327, 2717.9560547, -3579.6923828, 3499.1313477

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861244
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861245
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7356.8408203, 13242.6660156, -7216.5029297, 13038.7744141, -20395.6152344, 20459.1679688
1: -1426.1243896, 1876.9218750, -1404.0510254, 1842.2514648, -3268.3759766, 3280.9729004
2: -1058.1516113, 2178.4328613, -1039.4074707, 2136.3288574, -3194.4797363, 3217.8403320
3: -1136.0894775, 2916.8356934, -1111.8489990, 2869.4987793, -4005.5878906, 4028.6845703
4: -903.5424194, 2717.8261719, -886.5155640, 2665.7963867, -3569.3383789, 3604.3417969

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0856154, upper bound: 3295.0851637
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0856154, upper bound: 3295.0851637
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7300.9873047, 13135.7753906, -7367.8500977, 13288.1220703, -20589.1093750, 20503.6210938
1: -1415.4832764, 1860.1130371, -1430.2288818, 1874.8175049, -3290.3002930, 3290.3417969
2: -1049.0308838, 2161.0478516, -1058.9329834, 2177.8427734, -3226.8732910, 3219.9809570
3: -1127.1458740, 2893.9260254, -1134.5476074, 2927.2409668, -4054.3430176, 4028.4736328
4: -896.0032959, 2695.8010254, -903.5294189, 2717.3686523, -3613.3718262, 3599.3305664

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0861248, upper bound: 3295.0861248
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0861248, upper bound: 3295.0861247
time: 0.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.20 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840042
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0827318, upper bound: 3295.0839933
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840041
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0827272, upper bound: 3295.0839933
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840574
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842716
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840520
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0826765, upper bound: 3295.0842717
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808949
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808400
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808403
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0820927, upper bound: 3295.0808946
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0820347, upper bound: 3295.0808945
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0821048, upper bound: 3295.0808396
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0820339, upper bound: 3295.0808390
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0838737, upper bound: 3295.0845756
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0838737, upper bound: 3295.0845759
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0837044, upper bound: 3295.0810718
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0837044, upper bound: 3295.0810717
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0778949, upper bound: 3295.0823034
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0778949, upper bound: 3295.0823032
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0809542
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0809542
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0893820, upper bound: 3295.0851635
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0893820, upper bound: 3295.0851635
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861244
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0901097, upper bound: 3295.0861245
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0856154, upper bound: 3295.0851637
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0856154, upper bound: 3295.0851637
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0861248, upper bound: 3295.0861248
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -3295.0861248, upper bound: 3295.0861247

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6313.5898438, 11439.2998047, -6408.1044922, 11608.4726562, -17922.0625000, 17847.4042969
1: -1237.5677490, 1625.5245361, -1256.4428711, 1649.7174072, -2887.2846680, 2881.9665527
2: -914.5114746, 1883.4643555, -927.9998169, 1911.5996094, -2826.1110840, 2811.4641113
3: -977.6422729, 2513.3769531, -992.2892456, 2550.2851562, -3527.9274902, 3505.6657715
4: -779.6013184, 2350.7302246, -791.1828613, 2385.6437988, -3165.2451172, 3141.9130859

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840041
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840040
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7070.0253906, 12868.1845703, -6407.6245117, 11606.9257812, -18676.9453125, 19275.8085938
1: -1394.0129395, 1823.9219971, -1256.2086182, 1649.4316406, -3043.4440918, 3080.1306152
2: -1026.0909424, 2110.2644043, -927.8516235, 1911.5616455, -2937.6525879, 3038.1159668
3: -1097.9647217, 2823.7863770, -992.1035156, 2550.1325684, -3648.0971680, 3815.8898926
4: -875.2559204, 2634.1477051, -791.0579224, 2385.6154785, -3260.8713379, 3425.2053223

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839933
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839933
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6281.6176758, 11389.4179688, -6826.6230469, 12408.1142578, -18689.7324219, 18216.0390625
1: -1232.9003906, 1618.1141357, -1346.9033203, 1759.3177490, -2992.2182617, 2965.0170898
2: -910.4548950, 1874.5709229, -990.2504883, 2039.3680420, -2949.8229980, 2864.8212891
3: -973.3267212, 2501.6184082, -1058.8483887, 2723.7846680, -3697.1113281, 3560.4667969
4: -776.1359253, 2339.5175781, -844.5224609, 2544.4052734, -3320.5412598, 3184.0400391

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840040
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840040
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6993.3872070, 12738.4667969, -6812.5058594, 12382.5712891, -19375.9589844, 19550.9707031
1: -1380.6042480, 1805.1341553, -1344.1074219, 1755.6812744, -3136.2854004, 3149.2416992
2: -1015.6343994, 2087.9812012, -988.1917725, 2035.2017822, -3050.8354492, 3076.1728516
3: -1086.6717529, 2794.5678711, -1056.5832520, 2718.2968750, -3804.9680176, 3851.1511230
4: -866.2839355, 2606.3618164, -842.7504272, 2539.2238770, -3405.5073242, 3449.1115723

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827271, upper bound: 3295.0839933
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827271, upper bound: 3295.0839929
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6643.2602539, 12036.9667969, -6492.1191406, 11764.0644531, -18407.3222656, 18529.0859375
1: -1302.5325928, 1709.7119141, -1273.6243896, 1671.3344727, -2973.8666992, 2983.3364258
2: -961.8568115, 1981.7601318, -940.0939331, 1936.5493164, -2898.4062500, 2921.8537598
3: -1028.5734863, 2644.8769531, -1005.2426758, 2584.2661133, -3612.8395996, 3650.1193848
4: -820.1160278, 2473.2175293, -801.5236816, 2416.6115723, -3236.7272949, 3274.7412109

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840575
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826873, upper bound: 3295.0840574
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7414.2197266, 13492.8828125, -6494.1450195, 11767.0830078, -19181.3027344, 19987.0273438
1: -1461.6835938, 1912.0793457, -1273.8905029, 1671.7030029, -3133.3864746, 3185.9697266
2: -1075.6678467, 2212.8166504, -940.3137817, 1937.2546387, -3012.9223633, 3153.1303711
3: -1151.4097900, 2961.1921387, -1005.4588623, 2585.0976562, -3736.5073242, 3966.6506348
4: -917.6835938, 2761.9182129, -801.7158203, 2417.5019531, -3335.1855469, 3563.6337891

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6601.6914062, 11966.4902344, -6890.6225586, 12524.6787109, -19126.3710938, 18857.1093750
1: -1295.5000000, 1699.5073242, -1359.8105469, 1775.3552246, -3070.8552246, 3059.3171387
2: -956.1953735, 1969.8298340, -999.2584229, 2057.9379883, -3014.1333008, 2969.0881348
3: -1022.4171753, 2628.7824707, -1068.5084229, 2749.4404297, -3771.8576660, 3697.2910156
4: -815.2778931, 2458.2309570, -852.2350464, 2567.4914551, -3382.7692871, 3310.4655762

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840518
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840516
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7330.1088867, 13346.4970703, -6878.5161133, 12502.6435547, -19832.7519531, 20225.0097656
1: -1446.3873291, 1891.1153564, -1357.4023438, 1772.2279053, -3218.6152344, 3248.5173340
2: -1063.8919678, 2188.0866699, -997.4904175, 2054.3774414, -3118.2695312, 3185.5771484
3: -1138.6489258, 2928.4973145, -1066.5720215, 2744.7277832, -3883.3767090, 3995.0690918
4: -907.5806885, 2731.0676270, -850.7171631, 2563.0510254, -3470.6315918, 3581.7846680

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842717
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842717
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6278.7138672, 11373.2363281, -6741.1489258, 12187.5078125, -18466.2207031, 18114.3808594
1: -1229.7248535, 1615.6723633, -1318.2080078, 1731.1489258, -2960.8735352, 2933.8803711
2: -908.9979858, 1873.5208740, -974.5690918, 2008.0787354, -2917.0766602, 2848.0895996
3: -971.7885132, 2500.3549805, -1045.4226074, 2679.6757812, -3651.4643555, 3545.7773438
4: -774.8630981, 2338.6562500, -832.0000000, 2505.1635742, -3280.0261230, 3170.6560059

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808948
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808949
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6596.2934570, 11949.4062500, -6811.0585938, 12316.3642578, -18912.6582031, 18760.4648438
1: -1292.3282471, 1696.5979004, -1332.5867920, 1749.0068359, -3041.3349609, 3029.1845703
2: -954.5070801, 1968.0938721, -984.5441895, 2028.4621582, -2982.9692383, 2952.6381836
3: -1020.8444824, 2627.0412598, -1056.2000732, 2707.8435059, -3728.6875000, 3683.2412109
4: -813.7811279, 2456.4533691, -840.5238647, 2530.4853516, -3344.2661133, 3296.9772949

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6277.7333984, 11370.8623047, -7506.5117188, 13637.7207031, -19915.4531250, 18877.3750000
1: -1229.3737793, 1615.2603760, -1476.6550293, 1932.3079834, -3161.6816406, 3091.9155273
2: -908.7786255, 1873.3212891, -1087.6531982, 2237.7265625, -3146.5051270, 2960.9746094
3: -971.5260010, 2499.9846191, -1167.8000488, 2994.6296387, -3966.1555176, 3667.7844238
4: -774.6789551, 2338.4289551, -929.0763550, 2792.0947266, -3566.7729492, 3267.5053711

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808400
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824068, upper bound: 3295.0808400
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6597.8774414, 11951.8916016, -7573.7622070, 13761.3671875, -20359.2441406, 19525.6542969
1: -1292.5148926, 1696.8952637, -1490.5056152, 1949.5444336, -3242.0593262, 3187.4008789
2: -954.6849365, 1968.7209473, -1097.2950439, 2257.3173828, -3212.0024414, 3066.0158691
3: -1021.0154419, 2627.7287598, -1178.1555176, 3021.5952148, -4042.6105957, 3805.8842773
4: -813.9364014, 2457.2407227, -937.3016357, 2816.4816895, -3630.4179688, 3394.5422363

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808403
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808403
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6710.7958984, 12195.3388672, -6709.5537109, 12137.4531250, -18848.2500000, 18904.8925781
1: -1322.7301025, 1728.7535400, -1313.4820557, 1723.7219238, -3046.4509277, 3042.2355957
2: -973.1500244, 2005.2957764, -970.5620117, 1999.2607422, -2972.4106445, 2975.8579102
3: -1040.3549805, 2678.9963379, -1041.2230225, 2668.0742188, -3708.4289551, 3720.2187500
4: -829.8134766, 2502.4018555, -828.6105347, 2494.0810547, -3323.8942871, 3331.0122070

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820927, upper bound: 3295.0808946
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820927, upper bound: 3295.0808946
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6975.4384766, 12667.7353516, -6760.3051758, 12229.6396484, -19205.0781250, 19428.0390625
1: -1373.5803223, 1795.1315918, -1323.8389893, 1736.4241943, -3110.0043945, 3118.9707031
2: -1010.4437866, 2084.0612793, -977.6307983, 2013.7291260, -3024.1728516, 3061.6921387
3: -1080.7135010, 2783.4118652, -1048.9304199, 2688.3818359, -3769.0952148, 3832.3420410
4: -861.7261353, 2600.5046387, -834.6624756, 2512.0617676, -3373.7878418, 3435.1667480

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820347, upper bound: 3295.0808945
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820347, upper bound: 3295.0808946
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6695.9233398, 12168.5849609, -7433.4438477, 13512.0859375, -20208.0097656, 19602.0292969
1: -1319.7796631, 1724.9266357, -1463.7084961, 1914.0074463, -3233.7856445, 3188.6352539
2: -970.9813843, 2000.9299316, -1077.5089111, 2216.1394043, -3187.1208496, 3078.4389648
3: -1037.9597168, 2673.2385254, -1156.9448242, 2966.5864258, -4004.5458984, 3830.1828613
4: -827.9466553, 2496.9672852, -920.4192505, 2765.1823730, -3593.1289062, 3417.3864746

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821050, upper bound: 3295.0808395
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821050, upper bound: 3295.0808394
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6961.6308594, 12642.8583984, -7480.8461914, 13598.1123047, -20559.7421875, 20123.6953125
1: -1370.8563232, 1791.6036377, -1473.4305420, 1925.9460449, -3296.8022461, 3265.0341797
2: -1008.4513550, 2080.0397949, -1084.1590576, 2229.6853027, -3238.1367188, 3164.1987305
3: -1078.5390625, 2778.0683594, -1164.1252441, 2985.4270020, -4063.9653320, 3942.1931152
4: -860.0202637, 2595.4765625, -926.1014404, 2782.0554199, -3642.0754395, 3521.5776367

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820340, upper bound: 3295.0808390
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820340, upper bound: 3295.0808390
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7095.8320312, 12792.1064453, -6551.5952148, 11859.6337891, -18955.4589844, 19343.7011719
1: -1378.4302979, 1814.5434570, -1284.0604248, 1685.5675049, -3063.9975586, 3098.6035156
2: -1022.2173462, 2104.7377930, -947.9646606, 1953.5784912, -2975.7956543, 3052.7021484
3: -1095.2478027, 2816.7871094, -1014.3499756, 2606.2670898, -3701.5148926, 3831.1364746
4: -872.1802368, 2626.4162598, -808.3994751, 2437.5905762, -3309.7702637, 3434.8154297

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0835107, upper bound: 3295.0832363
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0834933, upper bound: 3295.0812803
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7981.7246094, 14372.6171875, -6551.5952148, 11859.6337891, -19841.3554688, 20915.3261719
1: -1544.4488525, 2027.5289307, -1284.0604248, 1685.5675049, -3230.0161133, 3311.5886230
2: -1145.3614502, 2351.5158691, -947.9646606, 1953.5784912, -3098.9399414, 3298.9140625
3: -1227.1889648, 3167.2612305, -1014.3499756, 2606.2670898, -3833.4558105, 4175.0566406
4: -977.3223267, 2934.3369141, -808.3994751, 2437.5905762, -3414.9128418, 3741.1308594

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0835107, upper bound: 3295.0832362
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0834933, upper bound: 3295.0812803
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7051.7529297, 12720.7070312, -6946.7788086, 12616.2197266, -19667.9707031, 19667.4804688
1: -1371.4708252, 1804.0874023, -1369.7463379, 1788.9422607, -3160.4128418, 3173.8330078
2: -1016.4417725, 2091.7968750, -1006.7315063, 2074.3256836, -3090.7670898, 3098.5283203
3: -1089.0438232, 2800.3850098, -1077.2611084, 2770.5595703, -3859.6035156, 3877.6459961
4: -867.2311401, 2610.2014160, -858.7850952, 2587.6474609, -3454.8784180, 3468.9865723

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0830759, upper bound: 3295.0810708
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0833439, upper bound: 3295.0808499
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7939.3803711, 14304.5996094, -6946.7788086, 12616.2197266, -20555.5996094, 21247.2695312
1: -1537.7083740, 2017.5747070, -1369.7463379, 1788.9422607, -3326.6506348, 3387.3205566
2: -1139.8532715, 2339.1208496, -1006.7315063, 2074.3256836, -3214.1784668, 3345.8522949
3: -1221.2080078, 3151.5183105, -1077.2611084, 2770.5595703, -3991.7675781, 4223.5087891
4: -972.6069946, 2918.7167969, -858.7850952, 2587.6474609, -3560.2543945, 3777.0617676

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0830758, upper bound: 3295.0810709
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0833439, upper bound: 3295.0808496
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7371.1640625, 13273.0478516, -6522.5830078, 11803.3476562, -19174.5117188, 19795.6269531
1: -1429.2026367, 1881.7998047, -1277.6307373, 1677.2480469, -3106.4504395, 3159.4306641
2: -1060.5643311, 2183.7568359, -943.3400269, 1945.2565918, -3005.8208008, 3127.0964355
3: -1138.4980469, 2923.4445801, -1009.5220337, 2595.4125977, -3733.9104004, 3932.9663086
4: -905.5479736, 2724.5551758, -804.4385376, 2427.3720703, -3332.9199219, 3528.9931641

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778948, upper bound: 3295.0823028
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0812351
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8224.3349609, 14790.3281250, -6522.5830078, 11803.3476562, -20027.6777344, 21301.3886719
1: -1588.6395264, 2087.5747070, -1277.6307373, 1677.2480469, -3265.8876953, 3365.2055664
2: -1179.0495605, 2421.6989746, -943.3400269, 1945.2565918, -3124.3059082, 3364.5297852
3: -1264.6092529, 3260.3081055, -1009.5220337, 2595.4125977, -3860.0214844, 4262.6191406
4: -1006.4776001, 3021.3364258, -804.4385376, 2427.3720703, -3433.8496094, 3824.4147949

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778948, upper bound: 3295.0823029
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0812351
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7333.7807617, 13212.9033203, -6924.0058594, 12570.3037109, -19904.0839844, 20136.9062500
1: -1423.4003906, 1872.9094238, -1364.5776367, 1782.1503906, -3205.5507812, 3237.4870605
2: -1055.6735840, 2172.6921387, -1002.9802246, 2067.7050781, -3123.3779297, 3175.6723633
3: -1133.2655029, 2909.5319824, -1073.3890381, 2762.1647949, -3895.4301758, 3982.9206543
4: -901.3681030, 2710.7104492, -855.5761108, 2579.5925293, -3480.9602051, 3566.2866211

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0809529
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0807934
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8184.6289062, 14725.0371094, -6924.0058594, 12570.3037109, -20754.9335938, 21641.7128906
1: -1582.1625977, 2078.1298828, -1364.5776367, 1782.1503906, -3364.3127441, 3442.7072754
2: -1173.8215332, 2409.8420410, -1002.9802246, 2067.7050781, -3241.5261230, 3412.8220215
3: -1259.0161133, 3245.2722168, -1073.3890381, 2762.1647949, -4021.1806641, 4312.6157227
4: -1002.0170898, 3006.4079590, -855.5761108, 2579.5925293, -3581.6088867, 3861.5966797

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0809529
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0807934
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7077.8095703, 12758.1982422, -7212.6440430, 13037.4267578, -20115.2324219, 19970.8417969
1: -1373.9797363, 1809.5152588, -1404.3387451, 1841.9594727, -3215.9392090, 3213.8540039
2: -1019.6013794, 2099.8515625, -1039.2562256, 2135.4116211, -3155.0129395, 3139.1076660
3: -1092.5825195, 2809.5273438, -1111.5086670, 2868.5427246, -3961.1252441, 3921.0361328
4: -869.9713745, 2620.4853516, -886.4045410, 2664.5659180, -3534.5371094, 3506.8898926

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0801396, upper bound: 3295.0806580
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0802890, upper bound: 3295.0786530
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7984.2050781, 14376.4726562, -7212.6440430, 13037.4267578, -21012.6152344, 21566.1308594
1: -1544.8437500, 2027.7264404, -1404.3387451, 1841.9594727, -3386.8032227, 3432.0651855
2: -1145.7019043, 2352.5393066, -1039.2562256, 2135.4116211, -3280.7192383, 3389.3464355
3: -1227.6367188, 3168.3225098, -1111.5086670, 2868.5427246, -4095.5629883, 4272.4477539
4: -977.6362305, 2935.7392578, -886.4045410, 2664.5659180, -3642.2021484, 3819.7668457

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0801396, upper bound: 3295.0819783
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0802890, upper bound: 3295.0803120
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7023.2651367, 12650.3896484, -7369.4453125, 13294.3896484, -20317.6542969, 20019.8320312
1: -1363.3148193, 1793.0849609, -1431.2568359, 1875.9439697, -3239.2587891, 3224.3415527
2: -1010.7441406, 2082.6520996, -1059.5142822, 2178.3088379, -3189.0529785, 3142.1662598
3: -1083.7772217, 2786.5019531, -1134.8846436, 2928.0871582, -4011.1711426, 3921.3867188
4: -862.5905762, 2598.7646484, -903.9960327, 2717.9560547, -3580.5461426, 3502.7607422

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0803898, upper bound: 3295.0802189
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0803757, upper bound: 3295.0778127
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7862.8242188, 14145.4687500, -7369.4453125, 13294.3896484, -21142.2558594, 21490.4960938
1: -1519.7436523, 1994.3865967, -1431.2568359, 1875.9439697, -3395.6875000, 3425.6433105
2: -1127.3419189, 2316.1591797, -1059.5142822, 2178.3088379, -3304.6042480, 3373.0095215
3: -1208.7662354, 3118.6232910, -1134.8846436, 2928.0871582, -4134.5366211, 4246.2719727
4: -962.2037354, 2890.2333984, -903.9960327, 2717.9560547, -3680.1596680, 3791.8859863

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0801395, upper bound: 3295.0803489
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0803757, upper bound: 3295.0779825
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7354.0898438, 13239.5761719, -7216.5029297, 13038.7744141, -20392.8632812, 20456.0781250
1: -1425.3983154, 1877.0270996, -1404.0510254, 1842.2514648, -3267.6499023, 3281.0778809
2: -1058.0822754, 2179.1384277, -1039.4074707, 2136.3288574, -3194.4111328, 3218.5454102
3: -1135.9842529, 2916.3725586, -1111.8489990, 2869.4987793, -4005.4829102, 4028.2216797
4: -903.4498901, 2718.8935547, -886.5155640, 2665.7963867, -3569.2458496, 3605.4091797

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0805398
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773584, upper bound: 3295.0785076
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8226.8144531, 14794.2207031, -7216.5029297, 13038.7744141, -21260.4804688, 21984.6953125
1: -1589.0993652, 2087.8530273, -1404.0510254, 1842.2514648, -3431.3508301, 3491.9038086
2: -1179.4489746, 2422.6623535, -1039.4074707, 2136.3288574, -3315.6853027, 3459.6132812
3: -1265.1055908, 3261.1508789, -1111.8489990, 2869.4987793, -4134.5712891, 4364.9135742
4: -1006.8320923, 3022.6320801, -886.5155640, 2665.7963867, -3672.6284180, 3906.9531250

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0813705
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773584, upper bound: 3295.0795316
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7293.0917969, 13123.0400391, -7367.8500977, 13288.1220703, -20581.2109375, 20490.8867188
1: -1413.0150146, 1858.8798828, -1430.2288818, 1874.8175049, -3287.8325195, 3289.1088867
2: -1048.2624512, 2160.2849121, -1058.9329834, 2177.8427734, -3226.1052246, 3219.2177734
3: -1126.2535400, 2891.3232422, -1134.5476074, 2927.2409668, -4053.3481445, 4025.8708496
4: -895.2953491, 2695.1044922, -903.5294189, 2717.3686523, -3612.6635742, 3598.6337891

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0801885
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773584, upper bound: 3295.0777794
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8103.5195312, 14561.7285156, -7367.8500977, 13288.1220703, -21380.5410156, 21901.3632812
1: -1563.7209473, 2053.8422852, -1430.2288818, 1874.8175049, -3438.5385742, 3484.0712891
2: -1160.7880859, 2385.8906250, -1058.9329834, 2177.8427734, -3337.8642578, 3442.1010742
3: -1245.9888916, 3210.8359375, -1134.5476074, 2927.2409668, -4171.4760742, 4337.2597656
4: -991.1578369, 2976.5690918, -903.5294189, 2717.3686523, -3708.5263672, 3877.8620605

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0803484
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0773583, upper bound: 3295.0779808
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.83 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840041
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840040
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839933
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839933
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840040
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840040
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827271, upper bound: 3295.0839933
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0827271, upper bound: 3295.0839929
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840575
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826873, upper bound: 3295.0840574
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840518
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840516
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842717
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842717
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808948
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808949
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808400
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824068, upper bound: 3295.0808400
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808403
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808403
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0820927, upper bound: 3295.0808946
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0820927, upper bound: 3295.0808946
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0820347, upper bound: 3295.0808945
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0820347, upper bound: 3295.0808946
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0821050, upper bound: 3295.0808395
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0821050, upper bound: 3295.0808394
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0820340, upper bound: 3295.0808390
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0820340, upper bound: 3295.0808390
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0835107, upper bound: 3295.0832363
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0834933, upper bound: 3295.0812803
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0835107, upper bound: 3295.0832362
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0834933, upper bound: 3295.0812803
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0830759, upper bound: 3295.0810708
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0833439, upper bound: 3295.0808499
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0830758, upper bound: 3295.0810709
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0833439, upper bound: 3295.0808496
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778948, upper bound: 3295.0823028
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0812351
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778948, upper bound: 3295.0823029
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778933, upper bound: 3295.0812351
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0809529
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0807934
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0809529
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0778930, upper bound: 3295.0807934
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0801396, upper bound: 3295.0806580
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0802890, upper bound: 3295.0786530
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0801396, upper bound: 3295.0819783
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0802890, upper bound: 3295.0803120
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0803898, upper bound: 3295.0802189
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0803757, upper bound: 3295.0778127
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0801395, upper bound: 3295.0803489
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0803757, upper bound: 3295.0779825
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0805398
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773584, upper bound: 3295.0785076
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0813705
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773584, upper bound: 3295.0795316
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0801885
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773584, upper bound: 3295.0777794
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773586, upper bound: 3295.0803484
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.83
Output dim: 3, lower bound: -3295.0773583, upper bound: 3295.0779808

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6313.5898438, 11439.2998047, -6315.1801758, 11443.6611328, -17757.2500000, 17754.4804688
1: -1237.5677490, 1625.5245361, -1238.2565918, 1625.9665527, -2863.5341797, 2863.7807617
2: -914.5114746, 1883.4643555, -914.6589966, 1884.1292725, -2798.6406250, 2798.1232910
3: -977.6422729, 2513.3769531, -977.7487183, 2514.1921387, -3491.8342285, 3491.1254883
4: -779.6013184, 2350.7302246, -779.7176514, 2351.6623535, -3131.2636719, 3130.4477539

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840041
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827699, upper bound: 3295.0840041
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6313.5898438, 11439.2998047, -7020.1674805, 12657.5556641, -18971.1445312, 18459.4667969
1: -1237.5677490, 1625.5245361, -1364.6799316, 1795.7382812, -3033.3061523, 2990.2043457
2: -914.5114746, 1883.4643555, -1011.6889648, 2081.3676758, -2995.8791504, 2895.1530762
3: -977.6422729, 2513.3769531, -1083.6900635, 2786.6857910, -3764.3278809, 3597.0668945
4: -779.6013184, 2350.7302246, -863.0711060, 2597.3776855, -3376.9790039, 3213.8012695

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840039
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827700, upper bound: 3295.0840042
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7070.0253906, 12868.1845703, -6315.1640625, 11443.0292969, -18513.0507812, 19183.3476562
1: -1394.0129395, 1823.9219971, -1238.1166992, 1625.8083496, -3019.8212891, 3062.0385742
2: -1026.0909424, 2110.2644043, -914.5859375, 1884.2282715, -2910.3193359, 3024.8500977
3: -1097.9647217, 2823.7863770, -977.6378784, 2514.2387695, -3612.2033691, 3801.4243164
4: -875.2559204, 2634.1477051, -779.6547241, 2351.8039551, -3227.0598145, 3413.8022461

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0764591, upper bound: 3295.0812212
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839931
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839931
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7070.0253906, 12868.1845703, -7020.4643555, 12657.3496094, -19727.3730469, 19888.6484375
1: -1394.0129395, 1823.9219971, -1364.5605469, 1795.6320801, -3189.6450195, 3188.4824219
2: -1026.0909424, 2110.2644043, -1011.6506348, 2081.5798340, -3107.6708984, 3121.9145508
3: -1097.9647217, 2823.7863770, -1083.6341553, 2786.8037109, -3884.7683105, 3907.4204102
4: -875.2559204, 2634.1477051, -863.0418091, 2597.6577148, -3472.9130859, 3497.1892090

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0764591, upper bound: 3295.0812212
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839931
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827319, upper bound: 3295.0839933
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6281.6176758, 11389.4179688, -6736.5590820, 12247.4648438, -18529.0820312, 18125.9726562
1: -1232.9003906, 1618.1141357, -1329.1041260, 1736.3442383, -2969.2446289, 2947.2180176
2: -910.4548950, 1874.5709229, -977.3162842, 2012.7979736, -2923.2529297, 2851.8872070
3: -973.3267212, 2501.6184082, -1044.6862793, 2688.5861816, -3661.9128418, 3546.3046875
4: -776.1359253, 2339.5175781, -833.3934326, 2511.5383301, -3287.6743164, 3172.9108887

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827669, upper bound: 3295.0840039
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827669, upper bound: 3295.0840040
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6281.6176758, 11389.4179688, -7403.9711914, 13395.3251953, -19676.9433594, 18793.3867188
1: -1232.9003906, 1618.1141357, -1448.6090088, 1896.6097412, -3129.5102539, 3066.7231445
2: -910.4548950, 1874.5709229, -1068.8953857, 2198.1362305, -3108.5910645, 2943.4663086
3: -973.3267212, 2501.6184082, -1144.3803711, 2946.8437500, -3920.1704102, 3645.9985352
4: -776.1359253, 2339.5175781, -911.9774780, 2742.5996094, -3518.7355957, 3251.4948730

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827669, upper bound: 3295.0840042
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827668, upper bound: 3295.0840042
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6993.3872070, 12738.4667969, -6722.8232422, 12222.6738281, -19216.0605469, 19461.2890625
1: -1380.6042480, 1805.1341553, -1326.3779297, 1732.8052979, -3113.4096680, 3131.5119629
2: -1015.6343994, 2087.9812012, -975.3115234, 2008.7612305, -3024.3955078, 3063.2927246
3: -1086.6717529, 2794.5678711, -1042.4805908, 2683.2673340, -3769.9384766, 3837.0483398
4: -866.2839355, 2606.3618164, -831.6675415, 2506.5183105, -3372.8017578, 3438.0290527

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827272, upper bound: 3295.0839931
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827272, upper bound: 3295.0839932
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6993.3872070, 12738.4667969, -7389.9648438, 13369.7871094, -20363.1738281, 20128.4316406
1: -1380.6042480, 1805.1341553, -1445.7803955, 1892.9681396, -3273.5722656, 3250.9145508
2: -1015.6343994, 2087.9812012, -1066.8330078, 2194.0664062, -3209.7001953, 3154.8142090
3: -1086.6717529, 2794.5678711, -1142.1407471, 2941.3720703, -4028.0432129, 3936.7084961
4: -866.2839355, 2606.3618164, -910.2107544, 2737.5454102, -3603.8291016, 3516.5725098

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827271, upper bound: 3295.0839930
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827272, upper bound: 3295.0839931
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6643.2602539, 12036.9667969, -6399.9233398, 11600.6699219, -18243.9257812, 18436.8886719
1: -1302.5325928, 1709.7119141, -1255.5965576, 1647.7590332, -2950.2915039, 2965.3083496
2: -961.8568115, 1981.7601318, -926.8560791, 1909.2890625, -2871.1459961, 2908.6162109
3: -1028.5734863, 2644.8769531, -990.8230591, 2548.4743652, -3577.0478516, 3635.6999512
4: -820.1160278, 2473.2175293, -790.1563110, 2382.8762207, -3202.9919434, 3263.3737793

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840577
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840577
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6643.2602539, 12036.9667969, -7102.2348633, 12808.5546875, -19451.8105469, 19139.2011719
1: -1302.5325928, 1709.7119141, -1381.3123779, 1816.7843018, -3119.3168945, 3091.0241699
2: -961.8568115, 1981.7601318, -1023.4472046, 2105.4938965, -3067.3505859, 3005.2072754
3: -1028.5734863, 2644.8769531, -1096.3531494, 2819.7331543, -3848.3066406, 3741.2297363
4: -820.1160278, 2473.2175293, -873.1379395, 2627.3513184, -3447.4665527, 3346.3554688

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840574
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826874, upper bound: 3295.0840576
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7414.2197266, 13492.8828125, -6402.3613281, 11604.5009766, -19018.7207031, 19895.2441406
1: -1461.6835938, 1912.0793457, -1255.9488525, 1648.2409668, -3109.9245605, 3168.0283203
2: -1075.6678467, 2212.8166504, -927.1427002, 1910.1219482, -2985.7895508, 3139.9592285
3: -1151.4097900, 2961.1921387, -991.1088867, 2549.4843750, -3700.8940430, 3952.3007812
4: -917.6835938, 2761.9182129, -790.4039917, 2383.9262695, -3301.6098633, 3552.3220215

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0765161, upper bound: 3295.0807342
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842716
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7414.2197266, 13492.8828125, -7105.2060547, 12813.1982422, -20227.4179688, 20598.0898438
1: -1461.6835938, 1912.0793457, -1381.7354736, 1817.3792725, -3279.0629883, 3293.8149414
2: -1075.6678467, 2212.8166504, -1023.8021240, 2106.4970703, -3182.1650391, 3236.6186523
3: -1151.4097900, 2961.1921387, -1096.7368164, 2820.9033203, -3972.3129883, 4057.9289551
4: -917.6835938, 2761.9182129, -873.4505615, 2628.6083984, -3546.2917480, 3635.3686523

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0765161, upper bound: 3295.0807342
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826803, upper bound: 3295.0842717
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6601.6914062, 11966.4902344, -6801.7006836, 12366.2031250, -18967.8945312, 18768.1835938
1: -1295.5000000, 1699.5073242, -1342.2298584, 1752.6846924, -3048.1845703, 3041.7360840
2: -956.1953735, 1969.8298340, -986.4880371, 2031.7354736, -2987.9309082, 2956.3178711
3: -1022.4171753, 2628.7824707, -1054.5205078, 2714.7080078, -3737.1252441, 3683.3027344
4: -815.2778931, 2458.2309570, -841.2485962, 2535.0744629, -3350.3522949, 3299.4794922

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840518
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840518
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6601.6914062, 11966.4902344, -7466.6186523, 13509.1552734, -20110.8476562, 19433.1074219
1: -1295.5000000, 1699.5073242, -1461.0957031, 1912.3566895, -3207.8566895, 3160.6027832
2: -956.1953735, 1969.8298340, -1077.7128906, 2216.2888184, -3172.4841309, 3047.5419922
3: -1022.4171753, 2628.7824707, -1153.8721924, 2971.8830566, -3994.3002930, 3782.6542969
4: -815.2778931, 2458.2309570, -919.5272827, 2765.2016602, -3580.4794922, 3377.7583008

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840519
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826842, upper bound: 3295.0840517
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7330.1088867, 13346.4970703, -6789.9008789, 12344.8056641, -19674.9140625, 20136.3964844
1: -1446.3873291, 1891.1153564, -1339.8822021, 1749.6394043, -3196.0268555, 3230.9973145
2: -1063.8919678, 2188.0866699, -984.7666626, 2028.2906494, -3092.1826172, 3172.8532715
3: -1138.6489258, 2928.4973145, -1052.6328125, 2710.1337891, -3848.7827148, 3981.1301270
4: -907.5806885, 2731.0676270, -839.7697144, 2530.7775879, -3438.3581543, 3570.8369141

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842717
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842716
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7330.1088867, 13346.4970703, -7454.9394531, 13487.7607422, -20817.8652344, 20801.4296875
1: -1446.3873291, 1891.1153564, -1458.7224121, 1909.3129883, -3355.7001953, 3349.8378906
2: -1063.8919678, 2188.0866699, -1075.9946289, 2212.9025879, -3276.7944336, 3264.0812988
3: -1138.6489258, 2928.4973145, -1152.0253906, 2967.3078613, -4105.9570312, 4080.5227051
4: -907.5806885, 2731.0676270, -918.0618286, 2760.9924316, -3668.5729980, 3649.1291504

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826765, upper bound: 3295.0842717
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826766, upper bound: 3295.0842717
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6278.7138672, 11373.2363281, -6640.4052734, 12008.1650391, -18286.8789062, 18013.6386719
1: -1229.7248535, 1615.6723633, -1298.3395996, 1705.4804688, -2935.2045898, 2914.0119629
2: -908.9979858, 1873.5208740, -960.1139526, 1978.1639404, -2887.1618652, 2833.6342773
3: -971.7885132, 2500.3549805, -1029.6723633, 2640.2685547, -3612.0571289, 3530.0270996
4: -774.8630981, 2338.6562500, -819.5803833, 2468.1254883, -3242.9880371, 3158.2363281

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808950
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808950
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6278.7138672, 11373.2363281, -7318.1953125, 13180.2929688, -19459.0078125, 18691.4316406
1: -1229.7248535, 1615.6723633, -1419.9822998, 1868.8093262, -3098.5339355, 3035.6547852
2: -908.9979858, 1873.5208740, -1053.3537598, 2166.8945312, -3075.8923340, 2926.8737793
3: -971.7885132, 2500.3549805, -1130.5506592, 2902.2062988, -3873.9948730, 3630.9057617
4: -774.8630981, 2338.6562500, -899.3317261, 2703.4350586, -3478.2973633, 3237.9880371

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808950
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808950
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6596.2934570, 11949.4062500, -6712.8759766, 12141.8134766, -18738.1074219, 18662.2812500
1: -1292.3282471, 1696.5979004, -1313.2386475, 1723.9649658, -3016.2932129, 3009.8361816
2: -954.5070801, 1968.0938721, -970.4583130, 1999.2950439, -2953.8022461, 2938.5522461
3: -1020.8444824, 2627.0412598, -1040.8469238, 2669.4575195, -3690.3015137, 3667.8879395
4: -813.7811279, 2456.4533691, -828.4288330, 2494.3688965, -3308.1496582, 3284.8820801

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6596.2934570, 11949.4062500, -7382.2509766, 13297.8222656, -19894.1152344, 19331.6562500
1: -1292.3282471, 1696.5979004, -1432.9873047, 1885.1104736, -3177.4384766, 3129.5852051
2: -954.5070801, 1968.0938721, -1062.4343262, 2185.4011230, -3139.9079590, 3030.5278320
3: -1020.8444824, 2627.0412598, -1140.3685303, 2927.8854980, -3948.7294922, 3767.4096680
4: -813.7811279, 2456.4533691, -907.0881348, 2726.4631348, -3540.2441406, 3363.5415039

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824079, upper bound: 3295.0808955
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6277.7333984, 11370.8623047, -7405.1494141, 13458.4833984, -19736.2148438, 18776.0117188
1: -1229.3737793, 1615.2603760, -1456.7558594, 1906.5969238, -3135.9702148, 3072.0161133
2: -908.7786255, 1873.3212891, -1073.1710205, 2207.6640625, -3116.4426270, 2946.4921875
3: -971.5260010, 2499.9846191, -1152.0220947, 2955.0551758, -3926.5810547, 3652.0063477
4: -774.6789551, 2338.4289551, -916.6416626, 2754.8659668, -3529.5446777, 3255.0705566

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824068, upper bound: 3295.0808399
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808400
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6277.7333984, 11370.8623047, -8069.3730469, 14604.6103516, -20882.3437500, 19440.2343750
1: -1229.3737793, 1615.2603760, -1575.9255371, 2066.1589355, -3295.5324707, 3191.1855469
2: -908.7786255, 1873.3212891, -1164.3256836, 2392.4748535, -3301.2534180, 3037.6469727
3: -971.5260010, 2499.9846191, -1250.0228271, 3211.3974609, -4182.9233398, 3750.0073242
4: -774.6789551, 2338.4289551, -994.4218750, 2985.2629395, -3759.9411621, 3332.8508301

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824068, upper bound: 3295.0808399
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824069, upper bound: 3295.0808400
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6597.8774414, 11951.8916016, -7475.0458984, 13587.1523438, -20185.0253906, 19426.9375000
1: -1292.5148926, 1696.8952637, -1471.1411133, 1924.4837646, -3216.9985352, 3168.0363770
2: -954.6849365, 1968.7209473, -1083.1932373, 2228.0515137, -3182.7363281, 3051.9140625
3: -1021.0154419, 2627.7287598, -1162.7855225, 2983.0988770, -4004.1142578, 3790.5141602
4: -813.9364014, 2457.2407227, -925.1923218, 2780.2495117, -3594.1860352, 3382.4331055

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0820016, upper bound: 3295.0808374
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0824049, upper bound: 3295.0808399
time: 0.93 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.62 + 416.52 = 421.13 seconds
