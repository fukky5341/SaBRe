## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2954094305


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5784860, 0.5784855)
1: (-16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8586702, 0.8586698)
2: (-4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6296978, 0.6296978)
3: (-12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6798048, 0.6798053)
4: (-10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5727429, 0.5727427)
5: (-7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6351314, 0.6351314)
6: (-5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9446011, 0.9446011)
7: (-11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8562999, 0.8562999)
8: (-2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5692954, 0.5692954)
9: (-2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6717248, 0.6717248)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.37 + 34.19 = 56.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2968933, upper bound: 0.2968939

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 149
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 554

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910394, upper bound: 0.2963058
time: 4.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968893, upper bound: 0.2968916
time: 9.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.62 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.62
Output dim: 0, lower bound: -0.2910394, upper bound: 0.2963058
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.62
Output dim: 0, lower bound: -0.2968893, upper bound: 0.2968916

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 10.2774029, 11.2591724, 10.2575979, 11.2680807, -0.5365911, 0.5449851
1: -16.7288456, -15.2674465, -16.7328491, -15.2654371, -0.8512936, 0.8492455
2: -4.6773100, -3.6597071, -4.6836247, -3.6549640, -0.6156864, 0.6126146
3: -12.7141876, -11.6603441, -12.7173424, -11.6536055, -0.6690178, 0.6662149
4: -10.3716249, -9.2244844, -10.3773499, -9.2131805, -0.5520501, 0.5485330
5: -7.7624745, -6.6608458, -7.7666683, -6.6591530, -0.6257734, 0.6289787
6: -5.3930507, -4.3150768, -5.4079771, -4.3095865, -0.9157438, 0.9252462
7: -11.3011446, -9.8888874, -11.3042183, -9.8824558, -0.8473425, 0.8436093
8: -2.8602200, -1.9520285, -2.8611026, -1.9507697, -0.5665379, 0.5661850
9: -2.4807153, -1.2665973, -2.4878013, -1.2591164, -0.6522708, 0.6533465

Time for backsubstitution: 20.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 912

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2881200, upper bound: 0.2963036
time: 5.42 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910369, upper bound: 0.2963041
time: 6.06 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 10.2391586, 11.2704563, 10.2391586, 11.2704544, -0.5462122, 0.5782003
1: -16.7365799, -15.2644939, -16.7365837, -15.2644939, -0.8634281, 0.8567891
2: -4.6894836, -3.6523602, -4.6894846, -3.6523633, -0.6258740, 0.6284761
3: -12.7181883, -11.6473265, -12.7181892, -11.6473255, -0.6794925, 0.6704516
4: -10.3790951, -9.2027006, -10.3790951, -9.2026939, -0.5720475, 0.5542607
5: -7.7704864, -6.6585093, -7.7704878, -6.6585097, -0.6332593, 0.6351318
6: -5.4215250, -4.3090539, -5.4215269, -4.3090539, -0.9232912, 0.9445992
7: -11.3050499, -9.8765717, -11.3050499, -9.8765707, -0.8562999, 0.8486710
8: -2.8618193, -1.9497166, -2.8618202, -1.9497166, -0.5712228, 0.5692945
9: -2.4918127, -1.2521224, -2.4918118, -1.2521186, -0.6711285, 0.6602845

Time for backsubstitution: 20.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2939738, upper bound: 0.2968903
time: 4.24 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968873, upper bound: 0.2968897
time: 6.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.39 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 0, lower bound: -0.2881200, upper bound: 0.2963036
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 0, lower bound: -0.2910369, upper bound: 0.2963041
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 0, lower bound: -0.2939738, upper bound: 0.2968903
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 31.39
Output dim: 0, lower bound: -0.2968873, upper bound: 0.2968897

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 10.2988224, 11.2546415, 10.2692413, 11.2677383, -0.5137606, 0.5238428
1: -16.7263870, -15.2686491, -16.7319622, -15.2659435, -0.8480549, 0.8468909
2: -4.6719208, -3.6735938, -4.6831751, -3.6626687, -0.6029406, 0.5985355
3: -12.7087107, -11.6744232, -12.7165699, -11.6613350, -0.6553559, 0.6514096
4: -10.3636560, -9.2276430, -10.3731899, -9.2133942, -0.5436077, 0.5399415
5: -7.7298355, -6.6714478, -7.7485633, -6.6596751, -0.5920541, 0.5990901
6: -5.3655801, -4.3246694, -5.3931475, -4.3102713, -0.8885632, 0.9000058
7: -11.2936878, -9.8963909, -11.3028564, -9.8866148, -0.8360276, 0.8349223
8: -2.8583870, -1.9606891, -2.8607922, -1.9550736, -0.5600839, 0.5566874
9: -2.4772348, -1.2703135, -2.4869478, -1.2611537, -0.6462600, 0.6488190

Time for backsubstitution: 20.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6143

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2881175, upper bound: 0.2948230
time: 4.08 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2881175, upper bound: 0.2963019
time: 6.23 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 10.2774086, 11.2591734, 10.2575998, 11.2680807, -0.5234871, 0.5408792
1: -16.7288456, -15.2674427, -16.7328491, -15.2654343, -0.8533058, 0.8479328
2: -4.6773090, -3.6597123, -4.6836252, -3.6549664, -0.6156850, 0.5997310
3: -12.7141876, -11.6603470, -12.7173376, -11.6536093, -0.6687036, 0.6531162
4: -10.3716249, -9.2244835, -10.3773479, -9.2131824, -0.5449775, 0.5483990
5: -7.7624717, -6.6608448, -7.7666674, -6.6591530, -0.5961719, 0.6264300
6: -5.3930464, -4.3150783, -5.4079752, -4.3095894, -0.8955803, 0.9225516
7: -11.3011427, -9.8888903, -11.3042183, -9.8824549, -0.8473401, 0.8385420
8: -2.8602185, -1.9520321, -2.8611031, -1.9507709, -0.5665364, 0.5633698
9: -2.4807153, -1.2665982, -2.4878018, -1.2591152, -0.6522214, 0.6498003

Time for backsubstitution: 20.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6143

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2910344, upper bound: 0.2948235
time: 5.82 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910344, upper bound: 0.2963015
time: 5.00 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 10.2608023, 11.2659683, 10.2508545, 11.2701225, -0.5231905, 0.5609460
1: -16.7340736, -15.2656841, -16.7356892, -15.2649899, -0.8601918, 0.8544030
2: -4.6840811, -3.6662459, -4.6890349, -3.6600661, -0.6131144, 0.6144085
3: -12.7127199, -11.6614838, -12.7174206, -11.6550808, -0.6658516, 0.6556401
4: -10.3711023, -9.2058563, -10.3749313, -9.2029085, -0.5635588, 0.5456357
5: -7.7377648, -6.6690989, -7.7523646, -6.6590281, -0.5995014, 0.6060674
6: -5.3939295, -4.3186274, -5.4066648, -4.3097315, -0.8959379, 0.9205961
7: -11.2976036, -9.8840714, -11.3036938, -9.8807316, -0.8450031, 0.8399906
8: -2.8599825, -1.9584780, -2.8615065, -1.9540443, -0.5647249, 0.5596991
9: -2.4883618, -1.2558613, -2.4909563, -1.2541592, -0.6651609, 0.6557298

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6143

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2939713, upper bound: 0.2954053
time: 6.09 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2939713, upper bound: 0.2968872
time: 4.18 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 10.2391644, 11.2704544, 10.2391586, 11.2704563, -0.5331247, 0.5779970
1: -16.7365799, -15.2644892, -16.7365799, -15.2644911, -0.8654466, 0.8554749
2: -4.6894808, -3.6523664, -4.6894855, -3.6523640, -0.6258726, 0.6155944
3: -12.7181883, -11.6473322, -12.7181883, -11.6473303, -0.6791744, 0.6572928
4: -10.3790903, -9.2026997, -10.3790932, -9.2026939, -0.5654840, 0.5541234
5: -7.7704873, -6.6585093, -7.7704873, -6.6585102, -0.6036582, 0.6336455
6: -5.4215231, -4.3090549, -5.4215255, -4.3090539, -0.9031296, 0.9445972
7: -11.3050451, -9.8765745, -11.3050470, -9.8765707, -0.8562989, 0.8436031
8: -2.8618202, -1.9497201, -2.8618207, -1.9497187, -0.5712218, 0.5664797
9: -2.4918120, -1.2521248, -2.4918098, -1.2521207, -0.6710651, 0.6567383

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6143
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6143

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968848, upper bound: 0.2954051
time: 10.17 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968847, upper bound: 0.2968871
time: 6.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 37.97 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2881175, upper bound: 0.2948230
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2881175, upper bound: 0.2963019
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2910344, upper bound: 0.2948235
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2910344, upper bound: 0.2963015
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2939713, upper bound: 0.2954053
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2939713, upper bound: 0.2968872
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2968848, upper bound: 0.2954051
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 37.97
Output dim: 0, lower bound: -0.2968847, upper bound: 0.2968871

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: 10.2988272, 11.2546425, 10.2692471, 11.2677374, -0.5136652, 0.5132680
1: -16.7263870, -15.2686481, -16.7319641, -15.2659435, -0.8488011, 0.8466668
2: -4.6719160, -3.6735926, -4.6831665, -3.6626692, -0.5994267, 0.5866189
3: -12.7087097, -11.6744251, -12.7165689, -11.6613379, -0.6521025, 0.6514087
4: -10.3636541, -9.2276449, -10.3731871, -9.2133951, -0.5407469, 0.5351942
5: -7.7298369, -6.6714606, -7.7485604, -6.6596947, -0.5615134, 0.5851090
6: -5.3655772, -4.3246689, -5.3931389, -4.3102713, -0.8885574, 0.8906193
7: -11.2936859, -9.8963909, -11.3028574, -9.8866138, -0.8364153, 0.8348026
8: -2.8583865, -1.9606912, -2.8607917, -1.9550748, -0.5557985, 0.5566587
9: -2.4772317, -1.2703152, -2.4869418, -1.2611532, -0.6462579, 0.6399426

Time for backsubstitution: 20.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5831

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2879733, upper bound: 0.2944378
time: 4.49 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2881168, upper bound: 0.2963000
time: 4.22 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 10.2774124, 11.2591705, 10.2576065, 11.2680788, -0.5233946, 0.5303159
1: -16.7288456, -15.2674446, -16.7328472, -15.2654362, -0.8540535, 0.8477077
2: -4.6773043, -3.6597123, -4.6836166, -3.6549661, -0.6109786, 0.5878148
3: -12.7141876, -11.6603489, -12.7173376, -11.6536102, -0.6654501, 0.6531143
4: -10.3716221, -9.2244835, -10.3773470, -9.2131796, -0.5420667, 0.5436513
5: -7.7624717, -6.6608577, -7.7666664, -6.6591740, -0.5656312, 0.6124418
6: -5.3930430, -4.3150797, -5.4079719, -4.3095884, -0.8955736, 0.9131618
7: -11.3011417, -9.8888893, -11.3042183, -9.8824539, -0.8477306, 0.8384213
8: -2.8602204, -1.9520335, -2.8611026, -1.9507735, -0.5622406, 0.5633440
9: -2.4807136, -1.2665992, -2.4877954, -1.2591171, -0.6516445, 0.6409240

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2891709, upper bound: 0.2961567
time: 5.64 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910330, upper bound: 0.2963005
time: 6.07 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 10.2608042, 11.2659674, 10.2508612, 11.2701235, -0.5231061, 0.5507185
1: -16.7340698, -15.2656813, -16.7356911, -15.2649908, -0.8609400, 0.8541770
2: -4.6840734, -3.6662447, -4.6890264, -3.6600659, -0.6087463, 0.6024919
3: -12.7127180, -11.6614866, -12.7174196, -11.6550827, -0.6625986, 0.6556387
4: -10.3710995, -9.2058582, -10.3749275, -9.2029076, -0.5635583, 0.5408878
5: -7.7377648, -6.6691113, -7.7523651, -6.6590500, -0.5689604, 0.5922794
6: -5.3939285, -4.3186297, -5.4066586, -4.3097324, -0.8959332, 0.9123135
7: -11.2976046, -9.8840723, -11.3036900, -9.8807325, -0.8453932, 0.8398728
8: -2.8599815, -1.9584789, -2.8615065, -1.9540474, -0.5604205, 0.5596766
9: -2.4883573, -1.2558620, -2.4909492, -1.2541595, -0.6651595, 0.6468525

Time for backsubstitution: 21.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5831

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2938210, upper bound: 0.2950165
time: 12.73 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2939702, upper bound: 0.2950166
time: 8.12 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 10.2437887, 11.2703276, 10.2472353, 11.2650003, -0.5225210, 0.5697389
1: -16.7355423, -15.2646103, -16.7344189, -15.2658806, -0.8624630, 0.8533821
2: -4.6837749, -3.6525402, -4.6794996, -3.6587038, -0.6121490, 0.6057463
3: -12.7177277, -11.6508656, -12.7143888, -11.6533604, -0.6727295, 0.6499000
4: -10.3756647, -9.2032080, -10.3732605, -9.2069473, -0.5582621, 0.5479815
5: -7.7699838, -6.6721649, -7.7554922, -6.6815329, -0.5801036, 0.6004894
6: -5.4152293, -4.3094897, -5.4104433, -4.3154931, -0.8908329, 0.9337187
7: -11.3038578, -9.8771410, -11.3026886, -9.8784695, -0.8529396, 0.8409138
8: -2.8613138, -1.9521494, -2.8583684, -1.9538553, -0.5667787, 0.5609174
9: -2.4870501, -1.2523525, -2.4833145, -1.2570798, -0.6613646, 0.6479278

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2952554
time: 4.51 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968829, upper bound: 0.2954042
time: 4.66 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 10.2391663, 11.2704544, 10.2391663, 11.2704573, -0.5330420, 0.5677907
1: -16.7365780, -15.2644901, -16.7365799, -15.2644901, -0.8661942, 0.8552499
2: -4.6894774, -3.6523674, -4.6894760, -3.6523643, -0.6203244, 0.6036777
3: -12.7181883, -11.6473322, -12.7181883, -11.6473312, -0.6759205, 0.6572909
4: -10.3790894, -9.2026987, -10.3790913, -9.2026939, -0.5654824, 0.5493774
5: -7.7704868, -6.6585226, -7.7704883, -6.6585340, -0.5731189, 0.6196523
6: -5.4215183, -4.3090549, -5.4215202, -4.3090534, -0.9031229, 0.9363165
7: -11.3050461, -9.8765726, -11.3050480, -9.8765726, -0.8566871, 0.8434834
8: -2.8618202, -1.9497206, -2.8618193, -1.9497204, -0.5669093, 0.5664582
9: -2.4918075, -1.2521217, -2.4918053, -1.2521212, -0.6710637, 0.6478615

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5831
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5831

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2967368
time: 6.71 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968829, upper bound: 0.2968860
time: 5.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.71 seconds
NS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2879733, upper bound: 0.2944378
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2881168, upper bound: 0.2963000
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2891709, upper bound: 0.2961567
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2910330, upper bound: 0.2963005
NS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2938210, upper bound: 0.2950165
NS_A2_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2939702, upper bound: 0.2950166
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2952554
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2968829, upper bound: 0.2954042
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2967368
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 33.71
Output dim: 0, lower bound: -0.2968829, upper bound: 0.2968860

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 10.2988300, 11.2546425, 10.2692499, 11.2677374, -0.5136638, 0.5023100
1: -16.7263870, -15.2686520, -16.7319641, -15.2659426, -0.8411322, 0.8466630
2: -4.6719160, -3.6735954, -4.6831670, -3.6626725, -0.5958359, 0.5866179
3: -12.7087107, -11.6744251, -12.7165670, -11.6613398, -0.6503320, 0.6509347
4: -10.3636541, -9.2276478, -10.3731861, -9.2133980, -0.5309744, 0.5351918
5: -7.7298341, -6.6714611, -7.7485614, -6.6596961, -0.5615132, 0.5790775
6: -5.3655758, -4.3246684, -5.3931398, -4.3102689, -0.8891287, 0.8896360
7: -11.2936821, -9.8963909, -11.3028488, -9.8866138, -0.8356252, 0.8263850
8: -2.8583870, -1.9606915, -2.8607917, -1.9550774, -0.5511265, 0.5566583
9: -2.4772305, -1.2703159, -2.4869447, -1.2611537, -0.6445231, 0.6394925

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2881168, upper bound: 0.2910323
time: 4.53 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2881168, upper bound: 0.2963000
time: 4.05 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 10.2870235, 11.2550211, 10.2631493, 11.2680168, -0.5136552, 0.5184135
1: -16.7240868, -15.2764912, -16.7324142, -15.2706394, -0.8445249, 0.8385410
2: -4.6744041, -3.6641588, -4.6831598, -3.6574783, -0.6050529, 0.5832438
3: -12.7130213, -11.6636724, -12.7172756, -11.6554375, -0.6616087, 0.6499033
4: -10.3677883, -9.2324162, -10.3772650, -9.2177505, -0.5317842, 0.5356731
5: -7.7565541, -6.6641121, -7.7632475, -6.6595287, -0.5593762, 0.6045213
6: -5.3917351, -4.3165536, -5.4074955, -4.3104324, -0.8926811, 0.9112792
7: -11.2920990, -9.8926678, -11.2990589, -9.8824940, -0.8388891, 0.8278346
8: -2.8579679, -1.9569635, -2.8609567, -1.9536114, -0.5571704, 0.5582700
9: -2.4791341, -1.2689703, -2.4874768, -1.2604923, -0.6477036, 0.6382551

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2891709, upper bound: 0.2908908
time: 7.27 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2891709, upper bound: 0.2961567
time: 9.10 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 10.2774153, 11.2591724, 10.2576084, 11.2680798, -0.5124993, 0.5263765
1: -16.7288513, -15.2674465, -16.7328472, -15.2654390, -0.8540516, 0.8400369
2: -4.6773038, -3.6597140, -4.6836166, -3.6549680, -0.6090214, 0.5847454
3: -12.7141857, -11.6603508, -12.7173386, -11.6536131, -0.6649771, 0.6513457
4: -10.3716221, -9.2244873, -10.3773460, -9.2131824, -0.5384926, 0.5339816
5: -7.7624698, -6.6608562, -7.7666645, -6.6591749, -0.5598001, 0.6100430
6: -5.3930464, -4.3150802, -5.4079709, -4.3095865, -0.8947411, 0.9132929
7: -11.3011322, -9.8888893, -11.3042116, -9.8824530, -0.8393159, 0.8376317
8: -2.8602195, -1.9520364, -2.8611026, -1.9507761, -0.5622411, 0.5586729
9: -2.4807146, -1.2666008, -2.4877942, -1.2591162, -0.6504183, 0.6391826

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 5826
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2910330, upper bound: 0.2910344
time: 6.80 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910330, upper bound: 0.2963002
time: 6.22 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 10.2437916, 11.2703276, 10.2472363, 11.2650003, -0.5116224, 0.5667574
1: -16.7355404, -15.2646151, -16.7344208, -15.2658844, -0.8624592, 0.8457117
2: -4.6837754, -3.6525424, -4.6794982, -3.6587040, -0.6102057, 0.6026778
3: -12.7177305, -11.6508656, -12.7143898, -11.6533585, -0.6722589, 0.6480970
4: -10.3756647, -9.2032080, -10.3732605, -9.2069502, -0.5574222, 0.5383077
5: -7.7699804, -6.6721611, -7.7554932, -6.6815319, -0.5742724, 0.5980945
6: -5.4152279, -4.3094878, -5.4104438, -4.3154926, -0.8899975, 0.9342947
7: -11.3038483, -9.8771439, -11.3026848, -9.8784695, -0.8444910, 0.8401237
8: -2.8613143, -1.9521520, -2.8583679, -1.9538581, -0.5667782, 0.5562458
9: -2.4870508, -1.2523525, -2.4833150, -1.2570806, -0.6609199, 0.6461606

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6143
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5831
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 5826
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6143

## Relational analysis of NS_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2954017, upper bound: 0.2954047
time: 7.74 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2954017, upper bound: 0.2954041
time: 7.72 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 10.2487907, 11.2663002, 10.2447109, 11.2703915, -0.5232894, 0.5564611
1: -16.7317734, -15.2735348, -16.7361450, -15.2696953, -0.8566599, 0.8460450
2: -4.6865649, -3.6568072, -4.6890168, -3.6548755, -0.6144037, 0.5991178
3: -12.7170200, -11.6506996, -12.7181263, -11.6491728, -0.6720853, 0.6540666
4: -10.3752460, -9.2106352, -10.3790064, -9.2072659, -0.5559982, 0.5413935
5: -7.7645664, -6.6617684, -7.7670693, -6.6588836, -0.5668638, 0.6117280
6: -5.4202085, -4.3105326, -5.4210458, -4.3099003, -0.9002151, 0.9344254
7: -11.2959738, -9.8803520, -11.2998810, -9.8766079, -0.8478227, 0.8328996
8: -2.8595605, -1.9546587, -2.8616714, -1.9525597, -0.5618391, 0.5613689
9: -2.4902105, -1.2545152, -2.4914851, -1.2535009, -0.6673589, 0.6451850

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5831
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 5826
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 149
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 5826
type: A, layer: 1, pos: 6143
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6183

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5831

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2950166
time: 7.22 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2950142, upper bound: 0.2967368
time: 7.44 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.56 + 553.58 = 610.14 seconds
