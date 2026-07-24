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
execution time: IAR + RelationalAnalysis = 21.36 + 35.97 = 57.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1361659, upper bound: 0.1361659

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 529

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 4653

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361623, upper bound: 0.1345627
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345626, upper bound: 0.1361623
time: 3.77 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.22 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.22
Output dim: 6, lower bound: -0.1361623, upper bound: 0.1345627
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.22
Output dim: 6, lower bound: -0.1345626, upper bound: 0.1361623

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2768662, 0.2748897
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3762922, 0.3787255
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2984805, 0.2973626
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4248714, 0.4274476
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2141315, 0.2154415
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2166770, 0.2185574
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2512414, 0.2499231
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2934244, 0.2952049
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3506615, 0.3501015
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3336830, 0.3333380

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 529

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 529

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1359699, upper bound: 0.1345617
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1361613, upper bound: 0.1343698
time: 4.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2748897, 0.2768664
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3787255, 0.3762920
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2973626, 0.2984805
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4274478, 0.4248714
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2154413, 0.2141315
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2185571, 0.2166767
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2499231, 0.2512414
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2952049, 0.2934244
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3501015, 0.3506615
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3333380, 0.3336830

Time for backsubstitution: 20.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 529

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 529

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1343702, upper bound: 0.1361613
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1345616, upper bound: 0.1359699
time: 4.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 6, lower bound: -0.1359699, upper bound: 0.1345617
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 6, lower bound: -0.1361613, upper bound: 0.1343698
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.47
Output dim: 6, lower bound: -0.1343702, upper bound: 0.1361613
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.47
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

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1314196, upper bound: 0.1325363
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1339473, upper bound: 0.1300106
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1316106, upper bound: 0.1323475
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1341360, upper bound: 0.1298199
time: 3.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1298198, upper bound: 0.1341360
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1323475, upper bound: 0.1316107
time: 4.81 seconds

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

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1300108, upper bound: 0.1339471
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1325363, upper bound: 0.1314194
time: 5.00 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.72 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1314196, upper bound: 0.1325363
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1339473, upper bound: 0.1300106
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1316106, upper bound: 0.1323475
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1341360, upper bound: 0.1298199
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1298198, upper bound: 0.1341360
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1323475, upper bound: 0.1316107
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1300108, upper bound: 0.1339471
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.72
Output dim: 6, lower bound: -0.1325363, upper bound: 0.1314194

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2761328, 0.2751908
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3769569, 0.3770795
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2947099, 0.2966504
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4256024, 0.4255362
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2145476, 0.2140507
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2169323, 0.2168080
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2511547, 0.2493424
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2917893, 0.2954829
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3503411, 0.3499596
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3329208, 0.3334060

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 401

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1336344, upper bound: 0.1297068
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1336413, upper bound: 0.1296999
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2768662, 0.2741404
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3747568, 0.3786144
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2936895, 0.2973626
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4231825, 0.4272251
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2131712, 0.2150108
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2160857, 0.2173990
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2512414, 0.2492181
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2924333, 0.2945607
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3506615, 0.3495014
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3330817, 0.3331769

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 401

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1338231, upper bound: 0.1295158
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1338300, upper bound: 0.1295090
time: 4.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2741406, 0.2771568
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3792791, 0.3745761
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2954323, 0.2936895
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4279556, 0.4220529
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2154272, 0.2123063
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2176545, 0.2150725
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2492181, 0.2522012
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2958128, 0.2927113
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3495014, 0.3515060
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3333046, 0.3331497

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 401

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1295088, upper bound: 0.1338301
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1295156, upper bound: 0.1338228
time: 4.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2748740, 0.2761064
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3770795, 0.3761108
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2944117, 0.2944016
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4255362, 0.4237418
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2140505, 0.2132663
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2168081, 0.2156637
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2493049, 0.2520769
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2964571, 0.2917893
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3498220, 0.3510478
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3334658, 0.3329208

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 401

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1296997, upper bound: 0.1336410
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1297066, upper bound: 0.1336342
time: 4.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.43 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1336344, upper bound: 0.1297068
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1336413, upper bound: 0.1296999
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1338231, upper bound: 0.1295158
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1338300, upper bound: 0.1295090
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1295088, upper bound: 0.1338301
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1295156, upper bound: 0.1338228
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1296997, upper bound: 0.1336410
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.43
Output dim: 6, lower bound: -0.1297066, upper bound: 0.1336342

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2759883, 0.2750740
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3767643, 0.3771658
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2978799, 0.2953036
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4243078, 0.4252238
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2088037, 0.2094018
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2106979, 0.2110425
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2510525, 0.2498631
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2890778, 0.2917655
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3448739, 0.3446350
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3130763, 0.3139143

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1326183, upper bound: 0.1288911
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1328154, upper bound: 0.1286985
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2760003, 0.2750621
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3769317, 0.3769987
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2974417, 0.2957418
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4250674, 0.4244642
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2094684, 0.2087373
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2100084, 0.2117320
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2510573, 0.2498584
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2890630, 0.2917802
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3447368, 0.3447721
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3140304, 0.3129601

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1326330, upper bound: 0.1288809
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1328256, upper bound: 0.1286838
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2767217, 0.2740235
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3745651, 0.3787005
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2968597, 0.2960162
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4218884, 0.4269121
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2074273, 0.2103618
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2098515, 0.2116336
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2511393, 0.2497388
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2897222, 0.2908435
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3451941, 0.3441768
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3132372, 0.3136854

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1328069, upper bound: 0.1287001
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1330041, upper bound: 0.1285076
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2767336, 0.2740116
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3747320, 0.3785336
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2964215, 0.2964544
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4226480, 0.4261522
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2080917, 0.2096974
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2091620, 0.2123232
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2511441, 0.2497340
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2897072, 0.2908585
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3450570, 0.3443139
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3141913, 0.3127310

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1328217, upper bound: 0.1286900
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1330142, upper bound: 0.1284928
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2740114, 0.2770505
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3791981, 0.3747323
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2967622, 0.2964215
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4268837, 0.4226480
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2101138, 0.2080919
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2125785, 0.2091619
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2497340, 0.2511815
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2908583, 0.2899849
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3443139, 0.3451951
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3127310, 0.3142593

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1284926, upper bound: 0.1330142
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1286897, upper bound: 0.1328218
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2740233, 0.2770386
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3793650, 0.3745651
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2963238, 0.2968597
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4276438, 0.4218881
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2107782, 0.2074274
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2118890, 0.2098515
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2497388, 0.2511767
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2908435, 0.2899997
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3441768, 0.3453321
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3136854, 0.3133051

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1285074, upper bound: 0.1330042
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1286998, upper bound: 0.1328069
time: 4.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1286835, upper bound: 0.1328251
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1288806, upper bound: 0.1326331
time: 6.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8300672, -2.2850757, -2.8300672, -2.2850757, -0.2747571, 0.2759881
1: -14.9170284, -14.1306601, -14.9170284, -14.1306601, -0.3771658, 0.3760998
2: -4.6656609, -4.2078738, -4.6656609, -4.2078738, -0.2953036, 0.2975724
3: -16.2423592, -15.5104733, -16.2423592, -15.5104733, -0.4252238, 0.4235764
4: -1.8037653, -1.2336226, -1.8037653, -1.2336226, -0.2094018, 0.2083874
5: -6.5716019, -6.1027842, -6.5716019, -6.1027842, -0.2110426, 0.2104427
6: 9.5077477, 10.1110182, 9.5077477, 10.1110182, -0.2498256, 0.2510525
7: -14.1340666, -13.4225636, -14.1340666, -13.4225636, -0.2914877, 0.2890778
8: -4.0802002, -3.3881569, -4.0802002, -3.3881569, -0.3444972, 0.3448737
9: -11.7132788, -10.8442383, -11.7132788, -10.8442383, -0.3138463, 0.3130763

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 761
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1495
type: DSZ, layer: 3, pos: 1828
type: DSZ, layer: 3, pos: 1986
type: DSZ, layer: 3, pos: 1206
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1504
type: DSZ, layer: 3, pos: 956
type: DSZ, layer: 3, pos: 1082

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 761

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1286983, upper bound: 0.1328155
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1288908, upper bound: 0.1326182
time: 5.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.01 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1326183, upper bound: 0.1288911
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1328154, upper bound: 0.1286985
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1326330, upper bound: 0.1288809
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1328256, upper bound: 0.1286838
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1328069, upper bound: 0.1287001
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1330041, upper bound: 0.1285076
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1328217, upper bound: 0.1286900
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1330142, upper bound: 0.1284928
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1284926, upper bound: 0.1330142
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1286897, upper bound: 0.1328218
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1285074, upper bound: 0.1330042
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1286998, upper bound: 0.1328069
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1286835, upper bound: 0.1328251
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1288806, upper bound: 0.1326331
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1286983, upper bound: 0.1328155
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.01
Output dim: 6, lower bound: -0.1288908, upper bound: 0.1326182

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 57.33 + 538.87 = 596.21 seconds
