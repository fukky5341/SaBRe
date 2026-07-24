## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38370851399999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3888612, 1.3888612)
1: (-8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7070894, 0.7070894)
2: (10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6135144, 0.6135144)
3: (-7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2651591, 1.2651591)
4: (-7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1643662, 1.1643662)
5: (-13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.3121305, 1.3121300)
6: (-12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5428610, 1.5428610)
7: (-5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2241964, 1.2241964)
8: (-3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9989872, 0.9989872)
9: (-5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2323804, 1.2323804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.48 + 35.79 = 58.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3915383, upper bound: 0.3915381

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: A, layer: 1, pos: 4629
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 6110

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909415
time: 7.32 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3915384
time: 6.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 13.95 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 13.95
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909415
NS_B2, status: Status.UNKNOWN, split count: 1, time: 13.95
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3915384

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -7.9975190, -6.0949278, -7.9841719, -6.1114225, -1.3644700, 1.3671460
1: -8.6655350, -7.4376101, -8.6630688, -7.4411459, -0.7019224, 0.7024815
2: 10.9188461, 11.8272009, 10.9245663, 11.8253231, -0.6087310, 0.6066878
3: -7.1249795, -5.6258416, -7.1201982, -5.6306152, -1.2576046, 1.2592249
4: -7.9744415, -6.4443436, -7.9626741, -6.4872665, -1.1062264, 1.1310384
5: -13.4263916, -11.7870398, -13.4101553, -11.7965965, -1.2948713, 1.2894335
6: -12.6894464, -10.7228622, -12.6830950, -10.7319241, -1.5291433, 1.5305243
7: -5.1234989, -3.6019237, -5.1188321, -3.6201439, -1.2003121, 1.2104588
8: -3.2838736, -2.1679921, -3.2815256, -2.1717944, -0.9938879, 0.9944539
9: -5.1186171, -3.6173499, -5.1076508, -3.6356163, -1.2055025, 1.2118673

Time for backsubstitution: 20.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6110
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 4629

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6110

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911994, upper bound: 0.3909406
time: 6.20 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909401
time: 6.33 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -7.9999943, -6.0890894, -7.9999909, -6.0890999, -1.3731675, 1.3888302
1: -8.6659203, -7.4363751, -8.6659222, -7.4363766, -0.7041106, 0.7070887
2: 10.9171724, 11.8272743, 10.9171734, 11.8272743, -0.6132891, 0.6138716
3: -7.1265326, -5.6245265, -7.1265302, -5.6245279, -1.2647018, 1.2661219
4: -7.9751201, -6.4292212, -7.9751205, -6.4292331, -1.1198440, 1.1639762
5: -13.4319868, -11.7855387, -13.4319754, -11.7855415, -1.3121290, 1.2963238
6: -12.6905022, -10.7197313, -12.6905041, -10.7197304, -1.5428548, 1.5476952
7: -5.1237650, -3.5955453, -5.1237631, -3.5955470, -1.2130084, 1.2232952
8: -3.2843947, -2.1668067, -3.2843943, -2.1668072, -0.9989839, 1.0035300
9: -5.1203327, -3.6108992, -5.1203327, -3.6109035, -1.2122574, 1.2323785

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6110
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 4629

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6110

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911994, upper bound: 0.3915376
time: 4.45 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3915366
time: 5.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.73 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 31.73
Output dim: 2, lower bound: -0.3911994, upper bound: 0.3909406
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 31.73
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909401
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 31.73
Output dim: 2, lower bound: -0.3911994, upper bound: 0.3915376
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 31.73
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3915366

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -7.9941859, -6.0949984, -7.9836020, -6.1114388, -1.3611150, 1.3664732
1: -8.6642914, -7.4376531, -8.6628571, -7.4411511, -0.7003667, 0.7019465
2: 10.9194422, 11.8271780, 10.9246674, 11.8253193, -0.6081498, 0.6065710
3: -7.1233025, -5.6259084, -7.1199112, -5.6306267, -1.2550969, 1.2583408
4: -7.9744134, -6.4483795, -7.9626684, -6.4879580, -1.1055069, 1.1269650
5: -13.4259529, -11.7872314, -13.4100742, -11.7966299, -1.2933254, 1.2881136
6: -12.6864738, -10.7229471, -12.6825848, -10.7319441, -1.5261970, 1.5299768
7: -5.1198664, -3.6022422, -5.1182117, -3.6201982, -1.1961393, 1.2091379
8: -3.2835460, -2.1691227, -3.2814713, -2.1719894, -0.9926214, 0.9921794
9: -5.1183224, -3.6190479, -5.1076016, -3.6359036, -1.2049952, 1.2101393

Time for backsubstitution: 21.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 4629

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3905978
time: 8.63 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909404
time: 8.12 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -7.9992914, -6.0873508, -7.9841700, -6.1114264, -1.3651710, 1.3745875
1: -8.6656113, -7.4347057, -8.6630697, -7.4411454, -0.7022061, 0.7050970
2: 10.9179878, 11.8288078, 10.9245682, 11.8253241, -0.6094761, 0.6083074
3: -7.1259584, -5.6215372, -7.1201982, -5.6306148, -1.2592058, 1.2631378
4: -7.9827948, -6.4429259, -7.9626741, -6.4872694, -1.1146398, 1.1310678
5: -13.4277878, -11.7842865, -13.4101553, -11.7965984, -1.2945509, 1.2936554
6: -12.6905823, -10.7169399, -12.6830921, -10.7319260, -1.5298281, 1.5364990
7: -5.1235933, -3.5931816, -5.1188316, -3.6201446, -1.2003312, 1.2185607
8: -3.2875881, -2.1675920, -3.2815261, -2.1717978, -0.9967594, 0.9965110
9: -5.1229963, -3.6173375, -5.1076517, -3.6356165, -1.2098231, 1.2116079

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 4629

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905979
time: 6.20 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909403
time: 7.87 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -7.9966578, -6.0891609, -7.9994226, -6.0891109, -1.3698068, 1.3881574
1: -8.6646776, -7.4364176, -8.6657085, -7.4363837, -0.7025547, 0.7065544
2: 10.9177704, 11.8272533, 10.9172745, 11.8272705, -0.6126966, 0.6137550
3: -7.1248579, -5.6245942, -7.1262431, -5.6245403, -1.2621899, 1.2652392
4: -7.9750934, -6.4332590, -7.9751148, -6.4299240, -1.1191249, 1.1599126
5: -13.4315481, -11.7857342, -13.4319019, -11.7855778, -1.3105822, 1.2950034
6: -12.6875286, -10.7198143, -12.6899910, -10.7197475, -1.5399084, 1.5471449
7: -5.1201296, -3.5958607, -5.1231413, -3.5956039, -1.2088356, 1.2219758
8: -3.2840638, -2.1679392, -3.2843375, -2.1670017, -0.9977179, 1.0012560
9: -5.1200409, -3.6125963, -5.1202812, -3.6111934, -1.2117510, 1.2306509

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 4629

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3912001
time: 4.57 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3915376
time: 4.70 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -8.0017624, -6.0815101, -7.9999866, -6.0890989, -1.3738656, 1.3962770
1: -8.6659985, -7.4334688, -8.6659203, -7.4363756, -0.7043948, 0.7097044
2: 10.9163179, 11.8288832, 10.9171734, 11.8272724, -0.6140072, 0.6154916
3: -7.1275139, -5.6202245, -7.1265287, -5.6245279, -1.2663059, 1.2700562
4: -7.9834776, -6.4278069, -7.9751205, -6.4292383, -1.1282654, 1.1640034
5: -13.4333801, -11.7827835, -13.4319763, -11.7855406, -1.3118081, 1.3005500
6: -12.6916399, -10.7138100, -12.6905012, -10.7197304, -1.5435448, 1.5536699
7: -5.1238599, -3.5867987, -5.1237631, -3.5955486, -1.2130284, 1.2314034
8: -3.2881131, -2.1664071, -3.2843933, -2.1668081, -1.0018649, 1.0055838
9: -5.1247139, -3.6108854, -5.1203308, -3.6109049, -1.2165799, 1.2321181

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110
type: A, layer: 1, pos: 4629

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3911989
time: 5.55 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3915376
time: 5.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.06 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3905978
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909404
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905979
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909403
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3912001
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3915376
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3911989
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3915376

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.9941859, -6.0949984, -7.9808340, -6.1114964, -1.3610291, 1.3637052
1: -8.6642914, -7.4376531, -8.6618271, -7.4411869, -0.7001364, 0.7006950
2: 10.9194422, 11.8271780, 10.9251566, 11.8253021, -0.6081347, 0.6060927
3: -7.1233025, -5.6259084, -7.1185160, -5.6306791, -1.2547054, 1.2563291
4: -7.9744134, -6.4483795, -7.9626474, -6.4913063, -1.1021409, 1.1269445
5: -13.4259529, -11.7872314, -13.4097099, -11.7967863, -1.2924271, 1.2869840
6: -12.6864738, -10.7229471, -12.6801157, -10.7320118, -1.5261583, 1.5275373
7: -5.1198664, -3.6022422, -5.1151991, -3.6204593, -1.1956210, 1.2057681
8: -3.2835460, -2.1691227, -3.2812071, -2.1729259, -0.9908657, 0.9914293
9: -5.1183224, -3.6190479, -5.1073589, -3.6373122, -1.2035928, 1.2099614

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2511
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 1090
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 1090
type: B, layer: 3, pos: 663
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.50 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B1_A1_B1_B1

### Relational analysis result of NS_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3752870
time: 5.81 seconds

## Relational analysis of NS_B1_A1_B1_B2

### Relational analysis result of NS_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3815089
time: 5.71 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.9941859, -6.0949984, -7.9859414, -6.1038542, -1.3685594, 1.3683009
1: -8.6642914, -7.4376531, -8.6631432, -7.4382448, -0.7030237, 0.7020190
2: 10.9194422, 11.8271780, 10.9236965, 11.8269310, -0.6097724, 0.6074951
3: -7.1233025, -5.6259084, -7.1211743, -5.6263132, -1.2590847, 1.2585812
4: -7.9744134, -6.4483795, -7.9710293, -6.4858484, -1.1070328, 1.1272714
5: -13.4259529, -11.7872314, -13.4115419, -11.7938528, -1.2952671, 1.2880068
6: -12.6864738, -10.7229471, -12.6842232, -10.7260094, -1.5321827, 1.5315409
7: -5.1198664, -3.6022422, -5.1189280, -3.6114116, -1.2043281, 1.2094793
8: -3.2835460, -2.1691227, -3.2852325, -2.1713867, -0.9924650, 0.9951839
9: -5.1183224, -3.6190479, -5.1120291, -3.6356018, -1.2053142, 1.2144952

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2511
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 1090
type: A, layer: 3, pos: 1090
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.47 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3756241
time: 6.72 seconds

## Relational analysis of NS_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3818494
time: 6.43 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.9992914, -6.0873508, -7.9808340, -6.1114964, -1.3656640, 1.3712530
1: -8.6656113, -7.4347057, -8.6618271, -7.4411869, -0.7014616, 0.7035887
2: 10.9179878, 11.8288078, 10.9251566, 11.8253021, -0.6095486, 0.6077304
3: -7.1259584, -5.6215372, -7.1185160, -5.6306791, -1.2569761, 1.2607126
4: -7.9827948, -6.4429259, -7.9626474, -6.4913063, -1.1105828, 1.1309686
5: -13.4277878, -11.7842865, -13.4097099, -11.7967863, -1.2934685, 1.2898278
6: -12.6905823, -10.7169399, -12.6801157, -10.7320118, -1.5301619, 1.5335608
7: -5.1235933, -3.5931816, -5.1151991, -3.6204593, -1.1993303, 1.2144985
8: -3.2875881, -2.1675920, -3.2812071, -2.1729259, -0.9946451, 0.9930267
9: -5.1229963, -3.6173375, -5.1073589, -3.6373122, -1.2081323, 1.2116818

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 1090
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 1090
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.59 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3814783, upper bound: 0.3752868
time: 7.07 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3824347, upper bound: 0.3815092
time: 7.44 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.9992914, -6.0873508, -7.9859414, -6.1038542, -1.3673120, 1.3699670
1: -8.6656113, -7.4347057, -8.6631432, -7.4382448, -0.7024522, 0.7030118
2: 10.9179878, 11.8288078, 10.9236965, 11.8269310, -0.6103096, 0.6082559
3: -7.1259584, -5.6215372, -7.1211743, -5.6263132, -1.2603211, 1.2618866
4: -7.9827948, -6.4429259, -7.9710293, -6.4858484, -1.1075521, 1.1313691
5: -13.4277878, -11.7842865, -13.4115419, -11.7938528, -1.2996435, 1.2941895
6: -12.6905823, -10.7169399, -12.6842232, -10.7260094, -1.5324650, 1.5338430
7: -5.1235933, -3.5931816, -5.1189280, -3.6114116, -1.2012992, 1.2114582
8: -3.2875881, -2.1675920, -3.2852325, -2.1713867, -0.9962354, 0.9968014
9: -5.1229963, -3.6173375, -5.1120291, -3.6356018, -1.2070317, 1.2133937

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2511
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 1090
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 1090
type: B, layer: 3, pos: 663
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.52 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3814784, upper bound: 0.3752868
time: 11.49 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3824348, upper bound: 0.3815090
time: 7.98 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.9966578, -6.0891609, -7.9966545, -6.0891705, -1.3697186, 1.3853889
1: -8.6646776, -7.4364176, -8.6646767, -7.4364185, -0.7023246, 0.7053039
2: 10.9177704, 11.8272533, 10.9177704, 11.8272514, -0.6126816, 0.6132739
3: -7.1248579, -5.6245942, -7.1248555, -5.6245956, -1.2618012, 1.2632184
4: -7.9750934, -6.4332590, -7.9750934, -6.4332705, -1.1157589, 1.1598892
5: -13.4315481, -11.7857342, -13.4315376, -11.7857361, -1.3096800, 1.2938757
6: -12.6875286, -10.7198143, -12.6875286, -10.7198181, -1.5398712, 1.5447073
7: -5.1201296, -3.5958607, -5.1201315, -3.5958669, -1.2083168, 1.2186050
8: -3.2840638, -2.1679392, -3.2840633, -2.1679406, -0.9959607, 1.0005069
9: -5.1200409, -3.6125963, -5.1200371, -3.6126008, -1.2103477, 1.2304692

Time for backsubstitution: 22.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 711
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2511
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 663
type: B, layer: 3, pos: 1090
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 1090
type: B, layer: 3, pos: 663
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: B, layer: 3, pos: 2140
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1202
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.64 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3758574
time: 6.10 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3820976
time: 8.05 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.9966578, -6.0891609, -8.0017624, -6.0815187, -1.3772602, 1.3900251
1: -8.6646776, -7.4364176, -8.6659975, -7.4334698, -0.7052135, 0.7066300
2: 10.9177704, 11.8272533, 10.9163208, 11.8288832, -0.6143196, 0.6146944
3: -7.1248579, -5.6245942, -7.1275101, -5.6202259, -1.2661829, 1.2654657
4: -7.9750934, -6.4332590, -7.9834752, -6.4278197, -1.1206493, 1.1683307
5: -13.4315481, -11.7857342, -13.4333687, -11.7827854, -1.3125305, 1.2948928
6: -12.6875286, -10.7198143, -12.6916399, -10.7138138, -1.5458946, 1.5487137
7: -5.1201296, -3.5958607, -5.1238594, -3.5868030, -1.2170405, 1.2223139
8: -3.2840638, -2.1679392, -3.2881126, -2.1664076, -0.9975572, 1.0042953
9: -5.1200409, -3.6125963, -5.1247120, -3.6108913, -1.2120676, 1.2350097

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 711
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2511
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 663
type: B, layer: 3, pos: 1090
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 1090
type: B, layer: 3, pos: 663
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: B, layer: 3, pos: 2140
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1202
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.48 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3761939
time: 5.85 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3824357
time: 6.33 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.0017624, -6.0815101, -7.9966545, -6.0891705, -1.3743563, 1.3929420
1: -8.6659985, -7.4334688, -8.6646767, -7.4364185, -0.7036510, 0.7081974
2: 10.9163179, 11.8288832, 10.9177704, 11.8272514, -0.6140802, 0.6149123
3: -7.1275139, -5.6202245, -7.1248555, -5.6245956, -1.2640705, 1.2676196
4: -7.9834776, -6.4278069, -7.9750934, -6.4332705, -1.1242089, 1.1647887
5: -13.4333801, -11.7827835, -13.4315376, -11.7857361, -1.3107204, 1.2967248
6: -12.6916399, -10.7138100, -12.6875286, -10.7198181, -1.5438786, 1.5507326
7: -5.1238599, -3.5867987, -5.1201315, -3.5958669, -1.2120271, 1.2273407
8: -3.2881131, -2.1664071, -3.2840633, -2.1679406, -0.9997492, 1.0021029
9: -5.1247139, -3.6108854, -5.1200371, -3.6126008, -1.2148890, 1.2321887

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 711
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 663
type: B, layer: 3, pos: 1090
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 1090
type: B, layer: 3, pos: 663
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: B, layer: 3, pos: 2140
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.50 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3814783, upper bound: 0.3758572
time: 5.99 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3824347, upper bound: 0.3820973
time: 8.13 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.0017624, -6.0815101, -8.0017624, -6.0815187, -1.3760152, 1.3916960
1: -8.6659985, -7.4334688, -8.6659975, -7.4334698, -0.7046351, 0.7076187
2: 10.9163179, 11.8288832, 10.9163208, 11.8288832, -0.6148407, 0.6154559
3: -7.1275139, -5.6202245, -7.1275101, -5.6202259, -1.2674017, 1.2688155
4: -7.9834776, -6.4278069, -7.9834752, -6.4278197, -1.1211743, 1.1653056
5: -13.4333801, -11.7827835, -13.4333687, -11.7827854, -1.3169074, 1.3010783
6: -12.6916399, -10.7138100, -12.6916399, -10.7138138, -1.5461807, 1.5510187
7: -5.1238599, -3.5867987, -5.1238594, -3.5868030, -1.2139945, 1.2242937
8: -3.2881131, -2.1664071, -3.2881126, -2.1664076, -1.0013213, 1.0058680
9: -5.1247139, -3.6108854, -5.1247120, -3.6108913, -1.2137880, 1.2339072

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: B, layer: 3, pos: 2132
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 711
type: A, layer: 3, pos: 711
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 2511
type: A, layer: 3, pos: 2511
type: B, layer: 3, pos: 1488
type: A, layer: 3, pos: 1488
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 2462
type: A, layer: 3, pos: 663
type: B, layer: 3, pos: 1090
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 1090
type: B, layer: 3, pos: 663
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 2462
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 661
type: B, layer: 3, pos: 661
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 226
type: B, layer: 3, pos: 226
type: B, layer: 3, pos: 212
type: A, layer: 3, pos: 212
type: B, layer: 3, pos: 2822
type: A, layer: 3, pos: 2822
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 2633
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: B, layer: 3, pos: 2140
type: A, layer: 3, pos: 2140
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1202
type: A, layer: 3, pos: 1202
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 2142
type: B, layer: 3, pos: 2142
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 1229
type: B, layer: 3, pos: 1229
type: B, layer: 3, pos: 2847
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: B, layer: 3, pos: 2380
type: B, layer: 3, pos: 3122
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.51 seconds

### Candidate
type: B, layer: 3, pos: 227

## Relational analysis of NS_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3814784, upper bound: 0.3758573
time: 6.08 seconds

## Relational analysis of NS_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3824348, upper bound: 0.3820977
time: 7.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 35.90 seconds
NS_B1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3752870
NS_B1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3815089
NS_B1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3756241
NS_B1_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3818494
NS_B1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3814783, upper bound: 0.3752868
NS_B1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3824347, upper bound: 0.3815092
NS_B1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3814784, upper bound: 0.3752868
NS_B1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3824348, upper bound: 0.3815090
NS_B2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3758574
NS_B2_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3820976
NS_B2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3811406, upper bound: 0.3761939
NS_B2_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3820981, upper bound: 0.3824357
NS_B2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3814783, upper bound: 0.3758572
NS_B2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3824347, upper bound: 0.3820973
NS_B2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3814784, upper bound: 0.3758573
NS_B2_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 35.90
Output dim: 2, lower bound: -0.3824348, upper bound: 0.3820977

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 58.27 + 510.45 = 568.72 seconds
