## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.133442582


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2867398, 0.2867398)
1: (-14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3827257, 0.3827260)
2: (-4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2954063, 0.2954063)
3: (-16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4403210, 0.4403207)
4: (-1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2216630, 0.2216629)
5: (-6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2256572, 0.2256573)
6: (9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2573330, 0.2573330)
7: (-14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.3041251, 0.3041251)
8: (-4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3534777, 0.3534777)
9: (-11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3350632, 0.3350632)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.95 + 34.67 = 57.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1361659, upper bound: 0.1361659

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 4653

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 529

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1359735, upper bound: 0.1361649
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361649, upper bound: 0.1359736
time: 3.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.94 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 6, lower bound: -0.1359735, upper bound: 0.1361649
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 6, lower bound: -0.1361649, upper bound: 0.1359736

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2860069, 0.2870572
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3833914, 0.3811915
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2957146, 0.2946944
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4410520, 0.4386325
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2220793, 0.2207029
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2259128, 0.2250663
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2572460, 0.2573704
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.3034809, 0.3044031
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3531573, 0.3536158
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3349028, 0.3351316

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4653

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4653

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1359699, upper bound: 0.1345617
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343702, upper bound: 0.1361613
time: 3.37 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2867398, 0.2860067
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3811913, 0.3827260
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2946944, 0.2954063
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4386325, 0.4403207
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2207029, 0.2216629
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2250664, 0.2256573
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2573330, 0.2572460
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.3041251, 0.3034809
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3534777, 0.3531573
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3350632, 0.3349028

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4653

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4653

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361613, upper bound: 0.1343698
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359699
time: 4.36 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.05 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.05
Output dim: 6, lower bound: -0.1359699, upper bound: 0.1345617
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.05
Output dim: 6, lower bound: -0.1343702, upper bound: 0.1361613
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.05
Output dim: 6, lower bound: -0.1361613, upper bound: 0.1343698
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.05
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359699

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2761328, 0.2752068
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3769569, 0.3771908
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2987885, 0.2966504
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4256024, 0.4257586
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2145476, 0.2144814
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2169323, 0.2179662
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2511547, 0.2499605
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2927802, 0.2954829
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3503411, 0.3502395
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3335221, 0.3334060

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 2132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1495

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1327521, upper bound: 0.1332943
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1347025, upper bound: 0.1313436
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2741563, 0.2771833
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3793902, 0.3747571
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2976708, 0.2977684
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4281783, 0.4231825
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2158577, 0.2131714
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2188127, 0.2160857
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2498363, 0.2512789
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2945609, 0.2937024
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3497810, 0.3507993
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3331769, 0.3337510

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 956

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1319784, upper bound: 0.1322046
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1302764, upper bound: 0.1336207
time: 4.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2768662, 0.2741563
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3747568, 0.3787255
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2977684, 0.2973626
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4231825, 0.4274476
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2131712, 0.2154415
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2160857, 0.2185574
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2512414, 0.2498362
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2934244, 0.2945607
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3506615, 0.3497810
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3336830, 0.3331769

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1986

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345598, upper bound: 0.1323673
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1341582, upper bound: 0.1327688
time: 4.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2748897, 0.2761331
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3771906, 0.3762920
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2966504, 0.2984805
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4257588, 0.4248714
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2144814, 0.2141315
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2179661, 0.2166767
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2499231, 0.2511547
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2952049, 0.2927802
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3501015, 0.3503411
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3333380, 0.3335221

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 956

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1341335, upper bound: 0.1312177
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1298085, upper bound: 0.1355425
time: 3.67 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.29 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1327521, upper bound: 0.1332943
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1347025, upper bound: 0.1313436
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1319784, upper bound: 0.1322046
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1302764, upper bound: 0.1336207
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1345598, upper bound: 0.1323673
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1341582, upper bound: 0.1327688
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1341335, upper bound: 0.1312177
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.29
Output dim: 6, lower bound: -0.1298085, upper bound: 0.1355425

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2682836, 0.2666914
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3654308, 0.3640151
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2944520, 0.2931640
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4193983, 0.4203300
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.1970375, 0.1968993
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2165546, 0.2175851
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2450014, 0.2424396
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2830074, 0.2866063
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3466623, 0.3450882
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3230903, 0.3228374

Time for backsubstitution: 21.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1986

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1331011, upper bound: 0.1300801
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1322953, upper bound: 0.1296373
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2596815, 0.2608123
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3523865, 0.3476522
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2909374, 0.2894657
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4192629, 0.4177504
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2030902, 0.1943699
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2131135, 0.2093041
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2396469, 0.2408464
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2472985, 0.2561895
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3456478, 0.3474147
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3235998, 0.3222852

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 2132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1828

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1297305, upper bound: 0.1335986
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1302545, upper bound: 0.1330747
time: 5.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2771785, 0.2740788
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3743060, 0.3782589
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2974854, 0.2973626
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4215713, 0.4258642
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2135802, 0.2158258
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2145973, 0.2170627
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2505349, 0.2489874
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2938716, 0.2950599
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3506525, 0.3497355
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3352296, 0.3346238

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1504

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1342126, upper bound: 0.1320167
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1342126, upper bound: 0.1320167
time: 4.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2767889, 0.2744684
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3742907, 0.3782742
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2977688, 0.2970791
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4215994, 0.4258358
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2135559, 0.2158500
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2145911, 0.2170689
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2503927, 0.2491298
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2939234, 0.2950082
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3506157, 0.3497720
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3351293, 0.3347244

Time for backsubstitution: 21.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 761

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1318101, upper bound: 0.1326618
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1340512, upper bound: 0.1302959
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2748528, 0.2761009
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3762035, 0.3757739
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2966356, 0.2980416
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4256382, 0.4248137
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2143765, 0.2140698
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2178440, 0.2166013
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2495846, 0.2505066
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2935710, 0.2919697
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3493016, 0.3490274
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3315899, 0.3321388

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1255

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1333914, upper bound: 0.1310130
time: 5.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1339279, upper bound: 0.1304699
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2748575, 0.2760961
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3766727, 0.3753047
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2962117, 0.2984655
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4257011, 0.4247510
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2144197, 0.2140265
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2178907, 0.2165544
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2492751, 0.2508160
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2943945, 0.2911460
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3487878, 0.3495412
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3319550, 0.3317740

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 401

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1294993, upper bound: 0.1352403
time: 4.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1295061, upper bound: 0.1352334
time: 3.40 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.98 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1331011, upper bound: 0.1300801
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1322953, upper bound: 0.1296373
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1297305, upper bound: 0.1335986
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1302545, upper bound: 0.1330747
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1342126, upper bound: 0.1320167
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1342126, upper bound: 0.1320167
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1318101, upper bound: 0.1326618
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1340512, upper bound: 0.1302959
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1333914, upper bound: 0.1310130
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1339279, upper bound: 0.1304699
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1294993, upper bound: 0.1352403
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 6, lower bound: -0.1295061, upper bound: 0.1352334

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2735062, 0.2768786
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3784633, 0.3739400
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2976849, 0.2977836
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4280107, 0.4227972
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2152394, 0.2125220
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2183570, 0.2154849
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2498571, 0.2513106
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2933359, 0.2922118
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3497930, 0.3508136
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3329570, 0.3335094

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1294336, upper bound: 0.1335764
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1297071, upper bound: 0.1332048
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2768147, 0.2741232
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3746753, 0.3785419
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2975910, 0.2971377
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4231281, 0.4274120
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2130618, 0.2153034
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2159846, 0.2185017
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2510791, 0.2496792
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2932332, 0.2943966
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3504803, 0.3494797
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3336825, 0.3331766

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 2138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1331555, upper bound: 0.1303692
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1325685, upper bound: 0.1309658
time: 4.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2768323, 0.2741041
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3745742, 0.3786390
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2975395, 0.2971854
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4231458, 0.4273932
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2130334, 0.2153294
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2160285, 0.2184563
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2510843, 0.2496678
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2932553, 0.2943697
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3503602, 0.3495932
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3336823, 0.3331769

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 766

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1495

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1308899, upper bound: 0.1303453
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1329452, upper bound: 0.1295389
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2228138, 0.2228665
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3538654, 0.3558786
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2936094, 0.2934232
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4251900, 0.4302363
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2087435, 0.2114333
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2139207, 0.2161551
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2280395, 0.2249974
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2896781, 0.2907927
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3507905, 0.3497055
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3355174, 0.3352602

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 766

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1315024, upper bound: 0.1266356
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1300934, upper bound: 0.1270128
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2691326, 0.2713108
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3554711, 0.3583074
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2804513, 0.2786851
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.3982551, 0.4001560
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2094957, 0.2093114
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.1889871, 0.1893498
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2422791, 0.2431142
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2735057, 0.2705942
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3479321, 0.3407907
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.2974553, 0.3036520

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1082
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1504

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1328768, upper bound: 0.1289982
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1322791, upper bound: 0.1291903
time: 5.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2747447, 0.2760003
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3769989, 0.3762670
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2957420, 0.2971342
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4244642, 0.4243362
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2087371, 0.2090520
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2117321, 0.2097530
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2498209, 0.2510573
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2915027, 0.2890630
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3446343, 0.3447366
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3128922, 0.3140304

Time for backsubstitution: 21.87 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.62 + 543.90 = 601.52 seconds
