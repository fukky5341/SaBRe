## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 1399.2956865315


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705)
1: (-86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016)
2: (-142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504)
3: (-159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043)
4: (-122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.87 + 1.73 = 4.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1420.6047579, upper bound: 1420.6047579

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.4361717, upper bound: 1420.5241769
time: 0.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.3990533, upper bound: 1420.3990533
time: 0.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.25 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 0, lower bound: -1420.4361717, upper bound: 1420.5241769
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 0, lower bound: -1420.3990533, upper bound: 1420.3990533

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -778.8613281, 817.1842651, -808.8602295, 845.3754272, -1624.2364502, 1626.0441895
1: -82.1220474, 57.9297447, -85.0590057, 60.2218590, -142.3439026, 142.9887085
2: -134.7200317, 150.7836609, -140.1623993, 156.0967560, -290.8167725, 290.9460449
3: -150.2254791, 95.9499817, -156.6241760, 99.4451447, -249.6706238, 252.5741425
4: -115.9597092, 123.8868408, -120.6713028, 128.1535645, -244.1132507, 244.5581207

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3255277
time: 0.58 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2988936, upper bound: 1420.3365910
time: 0.64 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -938.8359985, 992.7998657, -804.5632935, 841.9646606, -1780.8005371, 1797.3631592
1: -99.5204239, 70.0465088, -84.7160797, 59.8860779, -159.4064941, 154.7625427
2: -163.6974792, 182.7903595, -139.4566650, 155.4882507, -319.1857300, 322.2470093
3: -183.2069092, 116.3974380, -155.8790894, 98.9996414, -282.2065430, 272.2765198
4: -139.8352051, 150.1410065, -120.1676254, 127.6585236, -267.4937134, 270.3086243

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2494305, upper bound: 1420.2720089
time: 0.57 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2830722, upper bound: 1420.2830722
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.08 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3255277
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -1420.2988936, upper bound: 1420.3365910
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -1420.2494305, upper bound: 1420.2720089
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -1420.2830722, upper bound: 1420.2830722

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -722.4301758, 763.5773315, -628.6163940, 673.8399048, -1396.2700195, 1392.1937256
1: -76.5046692, 53.7708664, -67.2792358, 46.8195648, -123.3242340, 121.0501022
2: -124.9206009, 140.5813446, -109.2712860, 123.5667191, -248.4872894, 249.8526306
3: -138.9321747, 89.3632355, -121.7815704, 78.4632568, -217.3954315, 211.1448059
4: -107.1541519, 115.4477615, -93.8491287, 101.2286377, -208.3827667, 209.2968750

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.2001971
time: 0.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -778.8613281, 817.1842651, -767.3776245, 802.1433716, -1581.0042725, 1584.5614014
1: -82.1220474, 57.9297447, -80.7134171, 57.1139374, -139.2359619, 138.6431580
2: -134.7200317, 150.7836609, -132.5353088, 148.1486206, -282.8685913, 283.3189392
3: -150.2254791, 95.9499817, -148.1205597, 94.4083939, -244.6338806, 244.0705414
4: -115.9597092, 123.8868408, -114.9807892, 121.4931641, -237.4528656, 238.8676147

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2988936, upper bound: 1420.3365910
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2988936, upper bound: 1420.3365910
time: 0.54 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -894.1317749, 951.5374146, -626.8513794, 671.9837646, -1566.1154785, 1578.3887939
1: -95.2342224, 66.6191025, -67.0936508, 46.6969757, -141.9311676, 133.7127533
2: -155.8570862, 174.9238129, -108.9394531, 123.2536316, -279.1107178, 283.8632202
3: -174.1228485, 111.2263947, -121.4512329, 78.2468185, -252.3696289, 232.6776276
4: -132.9126892, 143.8392944, -93.6236115, 100.9728088, -233.8854828, 237.4629059

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4518614, upper bound: 1419.6431857
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4097473, upper bound: 1419.4865597
time: 0.51 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -938.8359985, 992.7998657, -766.7324829, 800.9290771, -1739.7647705, 1759.5323486
1: -99.5204239, 70.0465088, -80.6058731, 56.9886665, -156.5090637, 150.6523743
2: -163.6974792, 182.7903595, -132.5106964, 147.9754639, -311.6729431, 315.3009949
3: -183.2069092, 116.3974380, -148.3522339, 94.2645569, -277.4714661, 264.7496643
4: -139.8352051, 150.1410065, -115.1708832, 121.3308334, -261.1660156, 265.3118896

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2720089, upper bound: 1420.2494305
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2720089, upper bound: 1420.2830722
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.2001971
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1420.2988936, upper bound: 1420.3365910
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1420.2988936, upper bound: 1420.3365910
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1419.4518614, upper bound: 1419.6431857
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1419.4097473, upper bound: 1419.4865597
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1420.2720089, upper bound: 1420.2494305
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -1420.2720089, upper bound: 1420.2830722

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -641.3563843, 682.3488770, -613.5613403, 658.9370117, -1300.2934570, 1295.9100342
1: -68.0750580, 47.7173195, -65.7488937, 45.7027855, -113.7778473, 113.4662170
2: -110.3452225, 125.7435684, -106.6013565, 120.8432617, -231.1884613, 232.3448944
3: -122.7013016, 79.6037064, -118.8334274, 76.6914139, -199.3927155, 198.4371185
4: -95.5756836, 102.9958954, -91.7457352, 98.9776459, -194.5533295, 194.7416382

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6551701
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -647.5786133, 683.9223022, -603.1088257, 647.7906494, -1295.3690186, 1287.0311279
1: -68.4825058, 48.1623039, -64.6465836, 44.9654961, -113.4479980, 112.8088837
2: -111.1271133, 125.8816071, -104.5995865, 118.8131409, -229.9402466, 230.4812012
3: -123.3244858, 80.0291367, -116.4873962, 75.3918762, -198.7163696, 196.5165253
4: -96.2670593, 103.3016281, -90.0961304, 97.3186874, -193.5857544, 193.3977661

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.2001971
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -593.6575317, 641.0496826, -767.3776245, 802.1433716, -1395.8007812, 1408.4267578
1: -63.8927689, 44.1346626, -80.7134171, 57.1139374, -121.0066986, 124.8480606
2: -103.1070099, 117.4952927, -132.5353088, 148.1486206, -251.2556305, 250.0306091
3: -114.8406219, 74.4325714, -148.1205597, 94.4083939, -209.2490082, 222.5531158
4: -88.7581558, 96.3455963, -114.9807892, 121.4931641, -210.2513123, 211.3263855

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3365910
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3365910
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -741.4351196, 775.9283447, -767.3776245, 802.1433716, -1543.5784912, 1543.3057861
1: -78.0003128, 55.0136566, -80.7134171, 57.1139374, -135.1142273, 135.7270355
2: -127.8866882, 143.2390442, -132.5353088, 148.1486206, -276.0352783, 275.7743530
3: -142.9582672, 91.1946716, -148.1205597, 94.4083939, -237.3666687, 239.3152313
4: -111.1221695, 117.5245285, -114.9807892, 121.4931641, -232.6153259, 232.5052643

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3255277
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3255277
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -825.5070801, 884.5001831, -611.6477051, 656.9412842, -1482.4483643, 1496.1479492
1: -88.3024521, 61.5634842, -65.5497284, 45.5699654, -133.8723907, 127.1132126
2: -143.7265015, 162.6174774, -106.2357559, 120.5047226, -264.2312317, 268.8531799
3: -160.4211273, 103.1395111, -118.4626694, 76.4574890, -236.8786163, 221.6021729
4: -122.6249924, 133.6985626, -91.4904861, 98.7022400, -221.3272247, 225.1890564

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2656388, upper bound: 1419.5848604
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2656388, upper bound: 1419.5848604
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -795.6579590, 851.0986328, -601.3031006, 645.9520264, -1441.6098633, 1452.4017334
1: -85.0014801, 59.4608841, -64.4654160, 44.8575821, -129.8590393, 123.9263000
2: -138.5170135, 156.4363708, -104.2588196, 118.5009613, -257.0179138, 260.6950989
3: -154.6038513, 99.2697144, -116.1473541, 75.1730042, -229.7768402, 215.4170685
4: -118.2487411, 128.4304962, -89.8639755, 97.0674667, -215.3162079, 218.2944641

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2307376, upper bound: 1419.4329604
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2307376, upper bound: 1419.4865597
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -802.3933105, 859.4445190, -766.7324829, 800.9290771, -1603.3218994, 1626.1768799
1: -85.9452591, 59.6159515, -80.6058731, 56.9886665, -142.9339142, 140.2218170
2: -140.0189514, 157.7170563, -132.5106964, 147.9754639, -287.9944153, 290.2276917
3: -156.5053711, 100.2332001, -148.3522339, 94.2645569, -250.7699280, 248.5854187
4: -119.5028687, 129.6644135, -115.1708832, 121.3308334, -240.8336639, 244.8352661

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2494305
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2494305
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -888.4437866, 941.6607056, -766.7324829, 800.9290771, -1689.3728027, 1708.3931885
1: -94.3676300, 66.5027390, -80.6058731, 56.9886665, -151.3562622, 147.1086121
2: -154.4488373, 173.1627808, -132.5106964, 147.9754639, -302.4243164, 305.6734009
3: -172.5222168, 110.3946609, -148.3522339, 94.2645569, -266.7867432, 258.7468872
4: -132.0453339, 141.9928436, -115.1708832, 121.3308334, -253.3761444, 257.1636963

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2830722
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2830722
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.18 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6551701
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.2001971
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3365910
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3365910
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3255277
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2652519, upper bound: 1420.3255277
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.2656388, upper bound: 1419.5848604
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.2656388, upper bound: 1419.5848604
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.2307376, upper bound: 1419.4329604
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1419.2307376, upper bound: 1419.4865597
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2494305
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2494305
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2830722
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -1420.2383672, upper bound: 1420.2830722

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -641.3563843, 682.3488770, -578.9292603, 626.2942505, -1267.6506348, 1261.2778320
1: -68.0750580, 47.7173195, -62.3851929, 43.0680161, -111.1430740, 110.1025085
2: -110.3452225, 125.7435684, -100.4846649, 114.8308258, -225.1760559, 226.2282104
3: -122.7013016, 79.6037064, -111.9549713, 72.6892853, -195.3905945, 191.5586700
4: -95.5756836, 102.9958954, -86.7080994, 94.1160660, -189.6917419, 189.7039948

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -641.3563843, 682.3488770, -788.0588379, 845.2420044, -1486.5983887, 1470.4075928
1: -68.0750580, 47.7173195, -84.4896164, 58.5577545, -126.6327972, 132.2069244
2: -110.3452225, 125.7435684, -137.4958649, 155.0942078, -265.4393921, 263.2394409
3: -122.7013016, 79.6037064, -153.6787109, 98.5342102, -221.2355042, 233.2824097
4: -95.5756836, 102.9958954, -117.3579178, 127.5159225, -223.0916138, 220.3538208

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6551701
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6551701
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -647.5786133, 683.9223022, -567.7977295, 614.7230225, -1262.3015137, 1251.7199707
1: -68.4825058, 48.1623039, -61.2315521, 42.3133659, -110.7958679, 109.3938599
2: -111.1271133, 125.8816071, -98.3582764, 112.6833038, -223.8103943, 224.2398834
3: -123.3244858, 80.0291367, -109.4508133, 71.3347321, -194.6591949, 189.4799500
4: -96.2670593, 103.3016281, -84.9562988, 92.3891983, -188.6562500, 188.2578583

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1417389
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -647.5786133, 683.9223022, -774.6453247, 831.6268311, -1479.2054443, 1458.5675049
1: -68.4825058, 48.1623039, -83.1228180, 57.5725365, -126.0550385, 131.2851257
2: -111.1271133, 125.8816071, -135.1985168, 152.6077728, -263.7348328, 261.0801086
3: -123.3244858, 80.0291367, -151.1546478, 96.9024582, -220.2269287, 231.1837769
4: -96.2670593, 103.3016281, -115.4385986, 125.4268036, -221.6938629, 218.7402344

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1944546
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.2001971
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -593.6575317, 641.0496826, -741.4351196, 775.9283447, -1369.5859375, 1382.4846191
1: -63.8927689, 44.1346626, -78.0003128, 55.0136566, -118.9064102, 122.1349640
2: -103.1070099, 117.4952927, -127.8866882, 143.2390442, -246.3460541, 245.3819580
3: -114.8406219, 74.4325714, -142.9582672, 91.1946716, -206.0352936, 217.3908234
4: -88.7581558, 96.3455963, -111.1221695, 117.5245285, -206.2826385, 207.4677582

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3035030, upper bound: 1419.8128174
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -593.6575317, 641.0496826, -887.8563232, 941.1019287, -1534.7595215, 1528.9057617
1: -63.8927689, 44.1346626, -94.3063660, 66.4595108, -130.3522797, 138.4409943
2: -103.1070099, 117.4952927, -154.3389130, 173.0600891, -276.1671143, 271.8341980
3: -114.8406219, 74.4325714, -172.3881073, 110.3224564, -225.1630402, 246.8206635
4: -88.7581558, 96.3455963, -131.9457703, 141.9126892, -230.6708374, 228.2913361

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3035030, upper bound: 1419.8128174
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.2307376
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -741.4351196, 775.9283447, -741.4351196, 775.9283447, -1517.3635254, 1517.3635254
1: -78.0003128, 55.0136566, -78.0003128, 55.0136566, -133.0139465, 133.0139465
2: -127.8866882, 143.2390442, -127.8866882, 143.2390442, -271.1257019, 271.1257019
3: -142.9582672, 91.1946716, -142.9582672, 91.1946716, -234.1529388, 234.1529388
4: -111.1221695, 117.5245285, -111.1221695, 117.5245285, -228.6466522, 228.6466522

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.5310568, upper bound: 1418.7981743
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.3836009, upper bound: 1418.3897068
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -741.4351196, 775.9283447, -887.8563232, 941.1019287, -1682.5371094, 1663.7846680
1: -78.0003128, 55.0136566, -94.3063660, 66.4595108, -144.4598236, 149.3199615
2: -127.8866882, 143.2390442, -154.3389130, 173.0600891, -300.9467773, 297.5779419
3: -142.9582672, 91.1946716, -172.3881073, 110.3224564, -253.2807159, 263.5827637
4: -111.1221695, 117.5245285, -131.9457703, 141.9126892, -253.0348358, 249.4702454

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.5310568, upper bound: 1418.7981743
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.3836009, upper bound: 1418.3897068
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -825.5070801, 884.5001831, -577.7235107, 625.1989746, -1450.7060547, 1462.2236328
1: -88.3024521, 61.5634842, -62.2728806, 42.9857140, -131.2881470, 123.8363419
2: -143.7265015, 162.6174774, -100.2812653, 114.6334839, -258.3599548, 262.8987122
3: -160.4211273, 103.1395111, -111.7426376, 72.5550613, -232.9761963, 214.8821411
4: -122.6249924, 133.6985626, -86.5565720, 93.9558334, -216.5808258, 220.2551270

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -825.5070801, 884.5001831, -788.0588379, 845.2420044, -1670.7490234, 1672.5590820
1: -88.3024521, 61.5634842, -84.4896164, 58.5577545, -146.8601990, 146.0530548
2: -143.7265015, 162.6174774, -137.4958649, 155.0942078, -298.8207092, 300.1133423
3: -160.4211273, 103.1395111, -153.6787109, 98.5342102, -258.9553223, 256.8182373
4: -122.6249924, 133.6985626, -117.3579178, 127.5159225, -250.1409149, 251.0564880

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -795.6579590, 851.0986328, -566.6002808, 613.6317139, -1409.2895508, 1417.6989746
1: -85.0014801, 59.4608841, -61.1198044, 42.2319260, -127.2333908, 120.5806885
2: -138.5170135, 156.4363708, -98.1559830, 112.4859009, -251.0028992, 254.5923309
3: -154.6038513, 99.2697144, -109.2395477, 71.2011261, -225.8049622, 208.5092621
4: -118.2487411, 128.4304962, -84.8063812, 92.2301636, -210.4788513, 213.2368469

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3375966
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4329604
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -795.6579590, 851.0986328, -774.6453247, 831.6268311, -1627.2847900, 1625.7438965
1: -85.0014801, 59.4608841, -83.1228180, 57.5725365, -142.5740051, 142.5836945
2: -138.5170135, 156.4363708, -135.1985168, 152.6077728, -291.1247864, 291.6348267
3: -154.6038513, 99.2697144, -151.1546478, 96.9024582, -251.5063171, 250.4243622
4: -118.2487411, 128.4304962, -115.4385986, 125.4268036, -243.6755371, 243.8690796

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3903123
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4865597
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -802.3933105, 859.4445190, -741.4351196, 775.9283447, -1578.3216553, 1600.8795166
1: -85.9452591, 59.6159515, -78.0003128, 55.0136566, -140.9588776, 137.6162720
2: -140.0189514, 157.7170563, -127.8866882, 143.2390442, -283.2579956, 285.6037292
3: -156.5053711, 100.2332001, -142.9582672, 91.1946716, -247.7000427, 243.1914368
4: -119.5028687, 129.6644135, -111.1221695, 117.5245285, -237.0273438, 240.7865448

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.5055250, upper bound: 1418.6942041
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -802.3933105, 859.4445190, -887.6729736, 941.0486450, -1743.4416504, 1747.1174316
1: -85.9452591, 59.6159515, -94.3007812, 66.4441681, -152.3894348, 153.9167175
2: -140.0189514, 157.7170563, -154.3103790, 173.0446930, -313.0635681, 312.0274353
3: -156.5053711, 100.2332001, -172.3508453, 110.3148346, -266.8201904, 272.5840454
4: -119.5028687, 129.6644135, -131.9174957, 141.8984833, -261.4013367, 261.5819092

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.5055250, upper bound: 1418.6942041
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -888.4437866, 941.6607056, -741.4351196, 775.9283447, -1664.3720703, 1683.0958252
1: -94.3676300, 66.5027390, -78.0003128, 55.0136566, -149.3812408, 144.5030518
2: -154.4488373, 173.1627808, -127.8866882, 143.2390442, -297.6878662, 301.0494080
3: -172.5222168, 110.3946609, -142.9582672, 91.1946716, -263.7168579, 253.3529205
4: -132.0453339, 141.9928436, -111.1221695, 117.5245285, -249.5698395, 253.1150208

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2196320, upper bound: 1419.4258130
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -888.4437866, 941.6607056, -887.6729736, 941.0486450, -1829.4924316, 1829.3337402
1: -94.3676300, 66.5027390, -94.3007812, 66.4441681, -160.8117828, 160.8035126
2: -154.4488373, 173.1627808, -154.3103790, 173.0446930, -327.4935303, 327.4731140
3: -172.5222168, 110.3946609, -172.3508453, 110.3148346, -282.8370361, 282.7454834
4: -132.0453339, 141.9928436, -131.9174957, 141.8984833, -273.9438171, 273.9103394

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2196320, upper bound: 1419.5059947
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.14 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6551701
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6551701
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1417389
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1944546
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.2001971
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.3035030, upper bound: 1419.8128174
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.3035030, upper bound: 1419.8128174
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.2307376
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.5310568, upper bound: 1418.7981743
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.3836009, upper bound: 1418.3897068
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.5310568, upper bound: 1418.7981743
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.3836009, upper bound: 1418.3897068
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3375966
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4329604
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3903123
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4865597
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.5055250, upper bound: 1418.6942041
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.5055250, upper bound: 1418.6942041
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2196320, upper bound: 1419.4258130
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -1419.2196320, upper bound: 1419.5059947

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -578.9292603, 626.2942505, -1161.5648193, 1162.5341797
1: -57.9993362, 39.8736877, -62.3851929, 43.0680161, -101.0673523, 102.2588806
2: -92.6518936, 107.0407715, -100.4846649, 114.8308258, -207.4827271, 207.5254059
3: -103.4766312, 67.5666733, -111.9549713, 72.6892853, -176.1659241, 179.5216217
4: -80.6767960, 87.5416183, -86.7080994, 94.1160660, -174.7928009, 174.2497253

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -668.1575928, 694.5957642, -578.9292603, 626.2942505, -1294.4519043, 1273.5250244
1: -69.7120667, 49.2393341, -62.3851929, 43.0680161, -112.7800827, 111.6245117
2: -115.2760620, 128.4714050, -100.4846649, 114.8308258, -230.1068878, 228.9560547
3: -129.6795654, 81.5088272, -111.9549713, 72.6892853, -202.3688507, 193.4637756
4: -101.3218994, 105.0488129, -86.7080994, 94.1160660, -195.4379425, 191.7568817

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -788.0588379, 845.2420044, -1380.5124512, 1371.6638184
1: -57.9993362, 39.8736877, -84.4896164, 58.5577545, -116.5570679, 124.3632965
2: -92.6518936, 107.0407715, -137.4958649, 155.0942078, -247.7460938, 244.5366364
3: -103.4766312, 67.5666733, -153.6787109, 98.5342102, -202.0108337, 221.2453766
4: -80.6767960, 87.5416183, -117.3579178, 127.5159225, -208.1927032, 204.8995361

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -668.1575928, 694.5957642, -788.0588379, 845.2420044, -1513.3996582, 1482.6545410
1: -69.7120667, 49.2393341, -84.4896164, 58.5577545, -128.2698212, 133.7289429
2: -115.2760620, 128.4714050, -137.4958649, 155.0942078, -270.3702698, 265.9672546
3: -129.6795654, 81.5088272, -153.6787109, 98.5342102, -228.2137756, 235.1875305
4: -101.3218994, 105.0488129, -117.3579178, 127.5159225, -228.8378296, 222.4067383

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -567.7977295, 614.7230225, -1135.2495117, 1129.3818359
1: -55.9992790, 38.8239899, -61.2315521, 42.3133659, -98.3126450, 100.0555420
2: -89.5390854, 102.9940491, -98.3582764, 112.6833038, -202.2223663, 201.3522797
3: -99.4112320, 65.1949005, -109.4508133, 71.3347321, -170.7459259, 174.6456757
4: -77.8498535, 84.3572388, -84.9562988, 92.3891983, -170.2390442, 169.3135071

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1417389
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1417389
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -567.7977295, 614.7230225, -1291.9244385, 1269.6794434
1: -70.4951859, 49.7640610, -61.2315521, 42.3133659, -112.8085480, 110.9956131
2: -117.0030289, 129.8107605, -98.3582764, 112.6833038, -229.6862793, 228.1690216
3: -131.7602234, 82.5221405, -109.4508133, 71.3347321, -203.0949554, 191.9729462
4: -102.8403702, 106.2690353, -84.9562988, 92.3891983, -195.2295685, 191.2252808

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -774.6453247, 831.6268311, -1352.1533203, 1336.2291260
1: -55.9992790, 38.8239899, -83.1228180, 57.5725365, -113.5718079, 121.9468079
2: -89.5390854, 102.9940491, -135.1985168, 152.6077728, -242.1468506, 238.1925201
3: -99.4112320, 65.1949005, -151.1546478, 96.9024582, -196.3136597, 216.3495178
4: -77.8498535, 84.3572388, -115.4385986, 125.4268036, -203.2766571, 199.7958374

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.1944546
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.1944546
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -774.6453247, 831.6268311, -1508.8282471, 1476.5268555
1: -70.4951859, 49.7640610, -83.1228180, 57.5725365, -128.0677185, 132.8868713
2: -117.0030289, 129.8107605, -135.1985168, 152.6077728, -269.6107788, 265.0092468
3: -131.7602234, 82.5221405, -151.1546478, 96.9024582, -228.6626892, 233.6767731
4: -102.8403702, 106.2690353, -115.4385986, 125.4268036, -228.2671661, 221.7076416

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.2001971
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.2001971
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -725.6116333, 758.1499023, -1293.4204102, 1309.2166748
1: -57.9993362, 39.8736877, -76.2015686, 53.7488403, -111.7481613, 116.0752563
2: -92.6518936, 107.0407715, -125.1337585, 140.0160675, -232.6679688, 232.1745300
3: -103.4766312, 67.5666733, -140.1895142, 89.1001740, -192.5768127, 207.7561798
4: -80.6767960, 87.5416183, -109.0674667, 114.8309021, -195.5076599, 196.6090851

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -520.2152100, 561.2756958, -718.3186646, 749.5094604, -1269.7246094, 1279.5943604
1: -55.9684448, 38.7997932, -75.3157349, 53.0852890, -109.0537338, 114.1155243
2: -89.4835739, 102.9373322, -124.0058899, 138.4696350, -227.9531860, 226.9431915
3: -99.3523254, 65.1571655, -139.0906677, 88.0670929, -187.4194031, 204.2478333
4: -77.8049164, 84.3114014, -108.2220001, 113.5439224, -191.3487701, 192.5333710

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.2975492, upper bound: 1414.1976039
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.2079736, upper bound: 1413.8706496
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -871.5670166, 925.2544556, -1460.5247803, 1455.1718750
1: -57.9993362, 39.8736877, -92.6800766, 65.2474442, -123.2467728, 132.5537415
2: -92.6518936, 107.0407715, -151.4646149, 170.1179199, -262.7697754, 258.5053711
3: -103.4766312, 67.5666733, -169.1148682, 108.4088669, -211.8854980, 236.6815491
4: -80.6767960, 87.5416183, -129.4816132, 139.5145416, -220.1913147, 217.0232239

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.5325373, upper bound: 1416.3698207
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -520.2152100, 561.2756958, -857.9657593, 911.3911133, -1431.6063232, 1419.2413330
1: -55.9684448, 38.7997932, -91.2916412, 64.2508240, -120.2192612, 130.0914154
2: -89.4835739, 102.9373322, -149.0655823, 167.5783539, -257.0619202, 252.0028992
3: -99.3523254, 65.1571655, -166.4394379, 106.7488480, -206.1011658, 231.5966034
4: -77.8049164, 84.3114014, -127.4661026, 137.3717804, -215.1766510, 211.7774963

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -712.4851685, 745.6860352, -733.5927124, 767.6938477, -1480.1789551, 1479.2786865
1: -74.9797974, 52.8802986, -77.1787872, 54.4361877, -129.4159851, 130.0590820
2: -122.7741928, 137.5731201, -126.4807281, 141.6994476, -264.4736328, 264.0538330
3: -137.1078491, 87.5602570, -141.3334198, 90.2066879, -227.3145447, 228.8936768
4: -106.6139908, 112.8242569, -109.8683090, 116.2459030, -222.8598938, 222.6925659

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1412.4001780, upper bound: 1409.5134629
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1398.2973537, upper bound: 1398.8512546
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -741.2207642, 767.7214355, -715.5718994, 743.9064941, -1485.1271973, 1483.2933350
1: -77.4146347, 55.0076981, -74.8337021, 52.7917824, -130.2064056, 129.8413696
2: -127.9386063, 142.1416626, -123.6104050, 137.6546478, -265.5932617, 265.7520752
3: -143.6052856, 90.5608139, -138.8887787, 87.4940948, -231.0993652, 229.4495697
4: -111.5419006, 116.3920059, -108.1035004, 112.8383789, -224.3802795, 224.4955139

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.1196638, upper bound: 1400.1562194
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1397.9923019, upper bound: 1397.9923019
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -712.4851685, 745.6860352, -882.7396851, 936.6843262, -1649.1694336, 1628.4257812
1: -74.9797974, 52.8802986, -93.8419037, 66.0678482, -141.0476379, 146.7221985
2: -122.7741928, 137.5731201, -153.4687195, 172.2051697, -294.9793091, 291.0418396
3: -137.1078491, 87.5602570, -171.3752136, 109.7492371, -246.8570862, 258.9353943
4: -106.6139908, 112.8242569, -131.1689758, 141.2221069, -247.8360901, 243.9932251

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1412.3365156, upper bound: 1410.8004250
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.5442842, upper bound: 1406.4602521
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -741.2207642, 767.7214355, -847.6156006, 896.5361328, -1637.7568359, 1615.3370361
1: -77.4146347, 55.0076981, -89.9501266, 63.3873787, -140.8019714, 144.9578094
2: -127.9386063, 142.1416626, -147.3035583, 164.9686584, -292.9072266, 289.4452209
3: -143.6052856, 90.5608139, -164.6991730, 105.1542206, -248.7594452, 255.2599792
4: -111.5419006, 116.3920059, -126.1125641, 135.2911682, -246.8330688, 242.5045776

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1405.0688867, upper bound: 1400.1399643
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.9245533, upper bound: 1399.1995843
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -751.3156128, 809.6362915, -577.7235107, 625.1989746, -1376.5146484, 1387.3597412
1: -80.7691498, 55.9217567, -62.2728806, 42.9857140, -123.7548676, 118.1946411
2: -131.0220642, 148.6217194, -100.2812653, 114.6334839, -245.6555481, 248.9029388
3: -146.4423828, 94.2434235, -111.7426376, 72.5550613, -218.9974365, 205.9859924
4: -111.9889603, 122.0730820, -86.5565720, 93.9558334, -205.9447937, 208.6296539

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -809.5389404, 866.0775146, -577.7235107, 625.1989746, -1434.7379150, 1443.8010254
1: -86.5457306, 60.6105576, -62.2728806, 42.9857140, -129.5314484, 122.8834076
2: -140.5186310, 159.1654358, -100.2812653, 114.6334839, -255.1521149, 259.4467163
3: -156.6561127, 101.1267242, -111.7426376, 72.5550613, -229.2111664, 212.8693390
4: -120.2480698, 130.4785309, -86.5565720, 93.9558334, -214.2038727, 217.0350952

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -751.3156128, 809.6362915, -788.0588379, 845.2420044, -1596.5574951, 1597.6950684
1: -80.7691498, 55.9217567, -84.4896164, 58.5577545, -139.3269043, 140.4113617
2: -131.0220642, 148.6217194, -137.4958649, 155.0942078, -286.1162720, 286.1175842
3: -146.4423828, 94.2434235, -153.6787109, 98.5342102, -244.9765930, 247.9220734
4: -111.9889603, 122.0730820, -117.3579178, 127.5159225, -239.5048828, 239.4309998

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5323729
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5323729
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -809.5389404, 866.0775146, -788.0588379, 845.2420044, -1654.7810059, 1654.1363525
1: -86.5457306, 60.6105576, -84.4896164, 58.5577545, -145.1034851, 145.1001740
2: -140.5186310, 159.1654358, -137.4958649, 155.0942078, -295.6128235, 296.6613159
3: -156.6561127, 101.1267242, -153.6787109, 98.5342102, -255.1903076, 254.8053894
4: -120.2480698, 130.4785309, -117.3579178, 127.5159225, -247.7639618, 247.8364563

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5848604
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5848604
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -710.5509644, 764.5621338, -566.6002808, 613.6317139, -1324.1826172, 1331.1622314
1: -76.3381119, 52.9404297, -61.1198044, 42.2319260, -118.5700378, 114.0602264
2: -123.9077606, 140.3276520, -98.1559830, 112.4859009, -236.3936310, 238.4836426
3: -138.5691681, 88.9979706, -109.2395477, 71.2011261, -209.7702942, 198.2375031
4: -105.9541855, 115.1755905, -84.8063812, 92.2301636, -198.1843109, 199.9819641

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3375966
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3375966
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -787.9951172, 840.0554199, -566.6002808, 613.6317139, -1401.6268311, 1406.6552734
1: -84.1245575, 59.1402855, -61.1198044, 42.2319260, -126.3564835, 120.2600784
2: -136.5082550, 154.3729858, -98.1559830, 112.4859009, -248.9941406, 252.5289612
3: -152.1710205, 98.2243576, -109.2395477, 71.2011261, -223.3721313, 207.4638977
4: -117.2008057, 126.4473572, -84.8063812, 92.2301636, -209.4309540, 211.2537231

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4329604
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4329604
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -710.5509644, 764.5621338, -774.6453247, 831.6268311, -1542.1777344, 1539.2073975
1: -76.3381119, 52.9404297, -83.1228180, 57.5725365, -133.9106445, 136.0632477
2: -123.9077606, 140.3276520, -135.1985168, 152.6077728, -276.5155334, 275.5261841
3: -138.5691681, 88.9979706, -151.1546478, 96.9024582, -235.4716187, 240.1525726
4: -105.9541855, 115.1755905, -115.4385986, 125.4268036, -231.3809814, 230.6141968

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.3903123
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.3903123
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -787.9951172, 840.0554199, -774.6453247, 831.6268311, -1619.6219482, 1614.7004395
1: -84.1245575, 59.1402855, -83.1228180, 57.5725365, -141.6970978, 142.2631073
2: -136.5082550, 154.3729858, -135.1985168, 152.6077728, -289.1160278, 289.5714722
3: -152.1710205, 98.2243576, -151.1546478, 96.9024582, -249.0734863, 249.3789978
4: -117.2008057, 126.4473572, -115.4385986, 125.4268036, -242.6276093, 241.8859406

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4865597
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4865597
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -789.2792358, 848.5258179, -733.5927124, 767.6938477, -1556.9731445, 1582.1179199
1: -84.7729645, 58.6172256, -77.1787872, 54.4361877, -139.2091522, 135.7960052
2: -137.7963867, 155.5880127, -126.4807281, 141.6994476, -279.4958191, 282.0687256
3: -153.9386597, 98.8077621, -141.3334198, 90.2066879, -244.1453400, 240.1411743
4: -117.5238037, 127.9489136, -109.8683090, 116.2459030, -233.7697144, 237.8172150

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461750, upper bound: 1417.6538451
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -763.1124268, 809.9641724, -715.5718994, 743.9064941, -1507.0189209, 1525.5361328
1: -81.2185211, 56.6123772, -74.8337021, 52.7917824, -134.0102997, 131.4460602
2: -133.1019592, 148.9480743, -123.6104050, 137.6546478, -270.7565918, 272.5584717
3: -149.3598022, 94.7282486, -138.8887787, 87.4940948, -236.8538971, 233.6170197
4: -114.1255493, 122.3354645, -108.1035004, 112.8383789, -226.9639282, 230.4389648

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -789.2792358, 848.5258179, -882.6087036, 936.6659546, -1725.9450684, 1731.1343994
1: -84.7729645, 58.6172256, -93.8401031, 66.0565186, -150.8294830, 152.4573059
2: -137.7963867, 155.5880127, -153.4493561, 172.1967621, -309.9931335, 309.0373535
3: -153.9386597, 98.8077621, -171.3490601, 109.7461548, -263.6847839, 270.1567688
4: -117.5238037, 127.9489136, -131.1490936, 141.2137604, -258.7375488, 259.0979919

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -763.1124268, 809.9641724, -847.6853638, 896.6470947, -1659.7592773, 1657.6495361
1: -81.2185211, 56.6123772, -89.9637222, 63.3900719, -144.6085968, 146.5760956
2: -133.1019592, 148.9480743, -147.3189240, 164.9874268, -298.0893860, 296.2669983
3: -149.3598022, 94.7282486, -164.7167816, 105.1687088, -254.5285034, 259.4450378
4: -114.1255493, 122.3354645, -126.1251526, 135.3054352, -249.4309692, 248.4606171

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -818.7916870, 875.0894165, -725.6116333, 758.1499023, -1576.9412842, 1600.7010498
1: -87.4534149, 61.3306274, -76.2015686, 53.7488403, -141.2022552, 137.5321808
2: -142.1620789, 160.8739777, -125.1337585, 140.0160675, -282.1781006, 286.0077515
3: -158.5971375, 102.2414246, -140.1895142, 89.1001740, -247.6973114, 242.4309387
4: -121.6507797, 131.8574829, -109.0674667, 114.8309021, -236.4816589, 240.9249420

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1412.8793867, upper bound: 1411.1077048
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1398.4457422, upper bound: 1399.4980647
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -787.9951172, 840.0554199, -718.3186646, 749.5094604, -1537.5046387, 1558.3740234
1: -84.1245575, 59.1402855, -75.3157349, 53.0852890, -137.2098083, 134.4560089
2: -136.5082550, 154.3729858, -124.0058899, 138.4696350, -274.9779053, 278.3788147
3: -152.1710205, 98.2243576, -139.0906677, 88.0670929, -240.2380981, 237.3150330
4: -117.2008057, 126.4473572, -108.2220001, 113.5439224, -230.7447205, 234.6693115

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.1033365, upper bound: 1408.8713072
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1398.6042115, upper bound: 1399.8989334
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -818.7916870, 875.0894165, -872.0140381, 925.6750488, -1744.4663086, 1747.1032715
1: -87.4534149, 61.3306274, -92.7264252, 65.2802048, -152.7336121, 154.0570526
2: -142.1620789, 160.8739777, -151.5486755, 170.1953278, -312.3573303, 312.4226379
3: -158.5971375, 102.2414246, -169.2176666, 108.4632568, -267.0603943, 271.4591064
4: -121.6507797, 131.8574829, -129.5580292, 139.5747681, -261.2255249, 261.4154968

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.4995274, upper bound: 1395.2416480
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1394.4456632, upper bound: 1393.1562174
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -787.9951172, 840.0554199, -858.3973389, 911.7977905, -1699.7928467, 1698.4523926
1: -84.1245575, 59.1402855, -91.3362961, 64.2825165, -148.4070740, 150.4765778
2: -136.5082550, 154.3729858, -149.1463013, 167.6531982, -304.1614380, 303.5192566
3: -152.1710205, 98.2243576, -166.5374603, 106.8014221, -258.9724121, 264.7618103
4: -117.2008057, 126.4473572, -127.5390778, 137.4302521, -254.6310577, 253.9864197

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.3959801, upper bound: 1409.1459982
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1407.3178120, upper bound: 1407.3178120
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.40 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2659354, upper bound: 1419.6034692
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1417389
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1417389
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.1510365, upper bound: 1419.1474814
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.1944546
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.1944546
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.2001971
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3295656, upper bound: 1419.2001971
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1415.2975492, upper bound: 1414.1976039
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1414.2079736, upper bound: 1413.8706496
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.4329603, upper bound: 1419.2307376
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1412.4001780, upper bound: 1409.5134629
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.40
Output dim: 0, lower bound: -1398.2973537, upper bound: 1398.8512546
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1405.1196638, upper bound: 1400.1562194
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.40
Output dim: 0, lower bound: -1397.9923019, upper bound: 1397.9923019
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1412.3365156, upper bound: 1410.8004250
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1406.5442842, upper bound: 1406.4602521
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1405.0688867, upper bound: 1400.1399643
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1401.9245533, upper bound: 1399.1995843
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5848604
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5323729
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5323729
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5848604
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903122, upper bound: 1419.5848604
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3375966
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.3375966
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4329604
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.2117832, upper bound: 1419.4329604
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.3903123
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.3903123
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4865597
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4865597
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461750, upper bound: 1417.6538451
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1418.1461749, upper bound: 1417.6538451
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1412.8793867, upper bound: 1411.1077048
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1398.4457422, upper bound: 1399.4980647
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1413.1033365, upper bound: 1408.8713072
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1398.6042115, upper bound: 1399.8989334
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1401.4995274, upper bound: 1395.2416480
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.40
Output dim: 0, lower bound: -1394.4456632, upper bound: 1393.1562174
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1413.3959801, upper bound: 1409.1459982
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 0, lower bound: -1407.3178120, upper bound: 1407.3178120

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -535.2706299, 583.6050415, -1118.8753662, 1118.8753662
1: -57.9993362, 39.8736877, -57.9993362, 39.8736877, -97.8730240, 97.8730240
2: -92.6518936, 107.0407715, -92.6518936, 107.0407715, -199.6926575, 199.6926575
3: -103.4766312, 67.5666733, -103.4766312, 67.5666733, -171.0433044, 171.0433044
4: -80.6767960, 87.5416183, -80.6767960, 87.5416183, -168.2183990, 168.2183990

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1296146, upper bound: 1419.5065061
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1295561, upper bound: 1419.4601208
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -519.4495239, 560.5348511, -1095.8051758, 1103.0545654
1: -57.9993362, 39.8736877, -55.8911591, 38.7401924, -96.7395172, 95.7648392
2: -92.6518936, 107.0407715, -89.3350220, 102.8000107, -195.4519043, 196.3757935
3: -103.4766312, 67.5666733, -99.1783829, 65.0662689, -168.5429077, 166.7450409
4: -80.6767960, 87.5416183, -77.6812134, 84.2005844, -164.8773651, 165.2228394

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1296146, upper bound: 1419.5065061
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1295561, upper bound: 1419.4601208
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -668.1575928, 694.5957642, -535.2706299, 583.6050415, -1251.7625732, 1229.8663330
1: -69.7120667, 49.2393341, -57.9993362, 39.8736877, -109.5857544, 107.2386475
2: -115.2760620, 128.4714050, -92.6518936, 107.0407715, -222.3168182, 221.1232910
3: -129.6795654, 81.5088272, -103.4766312, 67.5666733, -197.2462311, 184.9854584
4: -101.3218994, 105.0488129, -80.6767960, 87.5416183, -188.8635254, 185.7255554

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0143744, upper bound: 1419.1082256
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1308619, upper bound: 1419.4501559
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -668.1575928, 694.5957642, -519.4495239, 560.5348511, -1228.6923828, 1214.0452881
1: -69.7120667, 49.2393341, -55.8911591, 38.7401924, -108.4522552, 105.1304703
2: -115.2760620, 128.4714050, -89.3350220, 102.8000107, -218.0760498, 217.8064270
3: -129.6795654, 81.5088272, -99.1783829, 65.0662689, -194.7458344, 180.6871796
4: -101.3218994, 105.0488129, -77.6812134, 84.2005844, -185.5224915, 182.7299957

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0143744, upper bound: 1419.1082256
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1308619, upper bound: 1419.4501559
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -749.1336060, 807.5985718, -1342.8690186, 1332.7384033
1: -57.9993362, 39.8736877, -80.5611496, 55.7510796, -113.7504120, 120.4348373
2: -92.6518936, 107.0407715, -130.6530914, 148.2424622, -240.8943481, 237.6938477
3: -103.4766312, 67.5666733, -146.0415344, 93.9885635, -197.4651794, 213.6082153
4: -80.6767960, 87.5416183, -111.6752853, 121.7673035, -202.4440613, 199.2169037

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3429364, upper bound: 1419.5559946
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3399676, upper bound: 1419.5202431
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -706.6948242, 761.0029907, -1296.2734375, 1290.2996826
1: -57.9993362, 39.8736877, -75.9761200, 52.6365929, -110.6359177, 115.8498077
2: -92.6518936, 107.0407715, -123.2560120, 139.6621094, -232.3139954, 230.2967529
3: -103.4766312, 67.5666733, -137.8598328, 88.5520630, -192.0286865, 205.4264984
4: -80.6767960, 87.5416183, -105.3963852, 114.6423111, -195.3190613, 192.9380035

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3429364, upper bound: 1419.5559946
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3399676, upper bound: 1419.5202431
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -668.1575928, 694.5957642, -749.1336060, 807.5985718, -1475.7561035, 1443.7293701
1: -69.7120667, 49.2393341, -80.5611496, 55.7510796, -125.4631500, 129.8004761
2: -115.2760620, 128.4714050, -130.6530914, 148.2424622, -263.5185242, 259.1244202
3: -129.6795654, 81.5088272, -146.0415344, 93.9885635, -223.6681061, 227.5503540
4: -101.3218994, 105.0488129, -111.6752853, 121.7673035, -223.0892029, 216.7240753

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2308158, upper bound: 1419.1631221
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3429653, upper bound: 1419.5128186
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -668.1575928, 694.5957642, -706.6948242, 761.0029907, -1429.1604004, 1401.2905273
1: -69.7120667, 49.2393341, -75.9761200, 52.6365929, -122.3486481, 125.2154388
2: -115.2760620, 128.4714050, -123.2560120, 139.6621094, -254.9381714, 251.7273865
3: -129.6795654, 81.5088272, -137.8598328, 88.5520630, -218.2316284, 219.3686523
4: -101.3218994, 105.0488129, -105.3963852, 114.6423111, -215.9642029, 210.4451752

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2308158, upper bound: 1419.1631221
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3429653, upper bound: 1419.5128186
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -535.2706299, 583.6050415, -1104.1314697, 1096.8544922
1: -55.9992790, 38.8239899, -57.9993362, 39.8736877, -95.8729630, 96.8233261
2: -89.5390854, 102.9940491, -92.6518936, 107.0407715, -196.5798645, 195.6459045
3: -99.4112320, 65.1949005, -103.4766312, 67.5666733, -166.9778595, 168.6715240
4: -77.8498535, 84.3572388, -80.6767960, 87.5416183, -165.3914642, 165.0340271

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.9032315, upper bound: 1418.5960864
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0242422, upper bound: 1419.0242419
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -520.2152100, 561.2756958, -1081.8020020, 1081.7993164
1: -55.9992790, 38.8239899, -55.9684448, 38.7997932, -94.7990723, 94.7924347
2: -89.5390854, 102.9940491, -89.4835739, 102.9373322, -192.4764099, 192.4775696
3: -99.4112320, 65.1949005, -99.3523254, 65.1571655, -164.5683441, 164.5471954
4: -77.8498535, 84.3572388, -77.8049164, 84.3114014, -162.1612396, 162.1621399

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.9032315, upper bound: 1418.5960864
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0242422, upper bound: 1419.0242419
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -535.2706299, 583.6050415, -1260.8063965, 1237.1522217
1: -70.4951859, 49.7640610, -57.9993362, 39.8736877, -110.3688736, 107.7633972
2: -117.0030289, 129.8107605, -92.6518936, 107.0407715, -224.0437775, 222.4626465
3: -131.7602234, 82.5221405, -103.4766312, 67.5666733, -199.3269043, 185.9987793
4: -102.8403702, 106.2690353, -80.6767960, 87.5416183, -190.3819580, 186.9458160

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.9064142, upper bound: 1418.6697732
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0150252, upper bound: 1419.0025200
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -520.2152100, 561.2756958, -1238.4770508, 1222.0969238
1: -70.4951859, 49.7640610, -55.9684448, 38.7997932, -109.2949829, 105.7325058
2: -117.0030289, 129.8107605, -89.4835739, 102.9373322, -219.9403381, 219.2943115
3: -131.7602234, 82.5221405, -99.3523254, 65.1571655, -196.9173889, 181.8744507
4: -102.8403702, 106.2690353, -77.8049164, 84.3114014, -187.1517334, 184.0739136

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.9064142, upper bound: 1418.6697732
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0150252, upper bound: 1419.0025200
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -749.1336060, 807.5985718, -1328.1248779, 1310.7174072
1: -55.9992790, 38.8239899, -80.5611496, 55.7510796, -111.7503586, 119.3851395
2: -89.5390854, 102.9940491, -130.6530914, 148.2424622, -237.7815552, 233.6470947
3: -99.4112320, 65.1949005, -146.0415344, 93.9885635, -193.3997345, 211.2364349
4: -77.8498535, 84.3572388, -111.6752853, 121.7673035, -199.6171265, 196.0325317

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1074778, upper bound: 1418.6563893
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2284885, upper bound: 1419.0845448
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -706.6948242, 761.0029907, -1281.5294189, 1268.2786865
1: -55.9992790, 38.8239899, -75.9761200, 52.6365929, -108.6358566, 114.8001099
2: -89.5390854, 102.9940491, -123.2560120, 139.6621094, -229.2012024, 226.2500000
3: -99.4112320, 65.1949005, -137.8598328, 88.5520630, -187.9632416, 203.0547180
4: -77.8498535, 84.3572388, -105.3963852, 114.6423111, -192.4921417, 189.7536316

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1074778, upper bound: 1418.6563893
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2284885, upper bound: 1419.0845448
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -749.1336060, 807.5985718, -1484.7999268, 1451.0152588
1: -70.4951859, 49.7640610, -80.5611496, 55.7510796, -126.2462616, 130.3252106
2: -117.0030289, 129.8107605, -130.6530914, 148.2424622, -265.2454834, 260.4638062
3: -131.7602234, 82.5221405, -146.0415344, 93.9885635, -225.7487793, 228.5636749
4: -102.8403702, 106.2690353, -111.6752853, 121.7673035, -224.6076202, 217.9443207

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1110674, upper bound: 1418.7346981
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2192715, upper bound: 1419.0628229
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -706.6948242, 761.0029907, -1438.2042236, 1408.5764160
1: -70.4951859, 49.7640610, -75.9761200, 52.6365929, -123.1317749, 125.7401810
2: -117.0030289, 129.8107605, -123.2560120, 139.6621094, -256.6651306, 253.0667267
3: -131.7602234, 82.5221405, -137.8598328, 88.5520630, -220.3122864, 220.3819733
4: -102.8403702, 106.2690353, -105.3963852, 114.6423111, -217.4826355, 211.6654205

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1110674, upper bound: 1418.7346983
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2192715, upper bound: 1419.0628229
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -671.3620605, 697.9876099, -1233.2583008, 1254.9670410
1: -57.9993362, 39.8736877, -70.0229797, 49.4841576, -107.4834900, 109.8966675
2: -92.6518936, 107.0407715, -115.8662262, 129.1412964, -221.7931824, 222.9069977
3: -103.4766312, 67.5666733, -130.4348145, 81.9095840, -185.3861847, 198.0014954
4: -80.6767960, 87.5416183, -101.8773499, 105.5730362, -186.2497864, 189.4189758

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.8582771, upper bound: 1418.5617595
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.8804133, upper bound: 1418.6202287
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -677.2014160, 701.8816528, -1237.1523438, 1260.8063965
1: -57.9993362, 39.8736877, -70.4951859, 49.7640610, -107.7633972, 110.3688736
2: -92.6518936, 107.0407715, -117.0030289, 129.8107605, -222.4626465, 224.0437775
3: -103.4766312, 67.5666733, -131.7602234, 82.5221405, -185.9987793, 199.3269043
4: -80.6767960, 87.5416183, -102.8403702, 106.2690353, -186.9458160, 190.3819580

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.8582771, upper bound: 1418.5617595
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.8804133, upper bound: 1418.6202287
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -512.8172607, 553.0322266, -678.3793335, 701.9313965, -1214.7486572, 1231.4116211
1: -55.1401634, 38.2292709, -70.5739441, 49.6910133, -104.8311768, 108.8032150
2: -88.0704193, 101.4315338, -117.3103256, 129.9945221, -218.0649261, 218.7418518
3: -97.7617645, 64.1899033, -132.2717896, 82.4946289, -180.2563934, 196.4616699
4: -76.6549530, 83.0828323, -103.1337051, 106.5165100, -183.1714630, 186.2165070

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.2079736, upper bound: 1413.8706496
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.2079736, upper bound: 1413.8706496
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -508.2808533, 549.7769775, -739.7254639, 768.8520508, -1277.1328125, 1289.5020752
1: -54.7515335, 37.9224167, -77.2273102, 54.3505020, -109.1020203, 115.1497269
2: -87.0902939, 100.8395615, -127.6042023, 142.1276245, -229.2179108, 228.4437561
3: -96.6174088, 63.7348022, -143.3038177, 90.2625275, -186.8798828, 207.0386200
4: -75.8831711, 82.5823898, -111.7117920, 116.6986465, -192.5818176, 194.2941895

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.2079736, upper bound: 1413.8706496
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.2079736, upper bound: 1413.8706496
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -818.7238770, 875.0343018, -1410.3046875, 1402.3288574
1: -57.9993362, 39.8736877, -87.4468002, 61.3255463, -119.3248749, 127.3204880
2: -92.6518936, 107.0407715, -142.1504669, 160.8632965, -253.5151825, 249.1912384
3: -103.4766312, 67.5666733, -158.5834198, 102.2333298, -205.7099609, 226.1500854
4: -80.6767960, 87.5416183, -121.6414032, 131.8480530, -212.5248413, 209.1830139

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4578761, upper bound: 1419.6523369
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4479124, upper bound: 1419.5849421
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -787.9951172, 840.0554199, -1375.3255615, 1371.6000977
1: -57.9993362, 39.8736877, -84.1245575, 59.1402855, -117.1396103, 123.9982452
2: -92.6518936, 107.0407715, -136.5082550, 154.3729858, -247.0248718, 243.5490265
3: -103.4766312, 67.5666733, -152.1710205, 98.2243576, -201.7009888, 219.7376709
4: -80.6767960, 87.5416183, -117.2008057, 126.4473572, -207.1241150, 204.7424316

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4578761, upper bound: 1419.6523369
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4479124, upper bound: 1419.5849421
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -520.2152100, 561.2756958, -818.7238770, 875.0343018, -1395.2495117, 1379.9995117
1: -55.9684448, 38.7997932, -87.4468002, 61.3255463, -117.2939758, 126.2465973
2: -89.4835739, 102.9373322, -142.1504669, 160.8632965, -250.3468475, 245.0877991
3: -99.3523254, 65.1571655, -158.5834198, 102.2333298, -201.5856323, 223.7405701
4: -77.8049164, 84.3114014, -121.6414032, 131.8480530, -209.6529388, 205.9527893

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1615127, upper bound: 1418.6682661
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3230761, upper bound: 1419.1035308
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -520.2152100, 561.2756958, -787.9951172, 840.0554199, -1360.2705078, 1349.2707520
1: -55.9684448, 38.7997932, -84.1245575, 59.1402855, -115.1087112, 122.9243469
2: -89.4835739, 102.9373322, -136.5082550, 154.3729858, -243.8565369, 239.4455872
3: -99.3523254, 65.1571655, -152.1710205, 98.2243576, -197.5766602, 217.3281708
4: -77.8049164, 84.3114014, -117.2008057, 126.4473572, -204.2521973, 201.5122070

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1615126, upper bound: 1418.6682661
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3230761, upper bound: 1419.1035308
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -694.8412476, 725.9603882, -661.9956665, 687.0663452, -1381.9075928, 1387.9560547
1: -72.9841995, 51.4866638, -68.9676285, 48.7798729, -121.7640686, 120.4542923
2: -119.6548004, 133.9630432, -114.2251129, 127.1471024, -246.8019104, 248.1881561
3: -133.8598175, 85.2448196, -128.5867157, 80.6537933, -214.5136108, 213.8315430
4: -104.1838226, 109.8342514, -100.4493713, 103.9453812, -208.1291656, 210.2836151

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1398.0164544, upper bound: 1398.0643909
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1398.0164544, upper bound: 1398.8512546
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -726.6294556, 752.4956665, -649.1353760, 670.4052124, -1397.0345459, 1401.6309814
1: -75.8171387, 53.8411865, -67.3735046, 47.5722694, -123.3894043, 121.2146912
2: -125.3734818, 139.3050232, -112.1906204, 124.2531204, -249.6265869, 251.4956360
3: -140.9382935, 88.7057037, -126.9202805, 78.7033844, -219.6416779, 215.6259613
4: -109.6452789, 114.0013046, -99.2067871, 101.5366287, -211.1818848, 213.2080536

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1402.1538590, upper bound: 1398.5383365
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.8876051, upper bound: 1396.5551982
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -694.8412476, 725.9603882, -811.6634521, 867.6263428, -1562.4675293, 1537.6237793
1: -72.9841995, 51.4866638, -86.7346802, 60.7835884, -133.7677917, 138.2213440
2: -119.6548004, 133.9630432, -140.9143677, 159.4931335, -279.1479492, 274.8773804
3: -133.8598175, 85.2448196, -157.1570892, 101.3750076, -235.2348328, 242.4019165
4: -104.1838226, 109.8342514, -120.5454254, 130.7808838, -234.9646759, 230.3796539

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.2808994, upper bound: 1405.7801215
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.2808994, upper bound: 1406.4602521
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -690.5667114, 720.4132080, -782.3241577, 835.0378418, -1525.6044922, 1502.7373047
1: -72.4124832, 51.0489120, -83.5888824, 58.7064781, -131.1189270, 134.6377716
2: -119.1253510, 133.0178833, -135.5315704, 153.4077606, -272.5330811, 268.5494385
3: -133.4870300, 84.5733566, -151.0366364, 97.5835571, -231.0705872, 235.6099548
4: -103.8977356, 109.0458069, -116.3374023, 125.6702499, -229.5679779, 225.3832092

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.2808994, upper bound: 1405.7801215
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1406.2808811, upper bound: 1406.4602286
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -726.6294556, 752.4956665, -782.9279175, 836.9470825, -1563.5765381, 1535.4233398
1: -75.8171387, 53.8411865, -83.6415863, 58.5927277, -134.4098663, 137.4827271
2: -125.3734818, 139.3050232, -135.9136658, 153.8886261, -279.2620850, 275.2185669
3: -140.9382935, 88.7057037, -151.8285675, 97.6967316, -238.6350250, 240.5342712
4: -109.6452789, 114.0013046, -116.8081360, 126.0397797, -235.6850433, 230.8094177

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1391.9570004, upper bound: 1391.9548562
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1392.0699672, upper bound: 1392.2128438
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -715.8914185, 737.5842896, -747.7206421, 795.0297852, -1510.9211426, 1485.3049316
1: -74.4365234, 52.8888054, -79.7671280, 56.0603447, -130.4968719, 132.6559143
2: -123.5221329, 136.7340088, -129.4424896, 146.2081451, -269.7302551, 266.1765137
3: -139.3221893, 87.0802307, -144.4313202, 92.9973450, -232.3195343, 231.5115509
4: -108.3990784, 111.9421768, -111.6100082, 119.7594299, -228.1585083, 223.5521851

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.59 + 415.46 = 420.05 seconds
