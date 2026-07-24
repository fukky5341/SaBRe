## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06767026200000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.8639135, -9.9779329, -10.8639135, -9.9779329, -0.2808142, 0.2808142)
1: (-6.5397010, -6.0593200, -6.5397010, -6.0593200, -0.2129111, 0.2129111)
2: (-8.3683348, -7.7006016, -8.3683348, -7.7006016, -0.2251027, 0.2251027)
3: (-2.2469616, -1.6020248, -2.2469616, -1.6020248, -0.2969913, 0.2969913)
4: (-7.7200961, -6.9018049, -7.7200961, -6.9018049, -0.2891202, 0.2891202)
5: (-8.0003510, -7.2460241, -8.0003510, -7.2460241, -0.4648643, 0.4648643)
6: (-13.4017906, -12.6399508, -13.4017906, -12.6399508, -0.3328521, 0.3328521)
7: (5.5834103, 5.9946704, 5.5834103, 5.9946704, -0.1187847, 0.1187847)
8: (-2.0689983, -1.3881192, -2.0689983, -1.3881192, -0.2629046, 0.2629046)
9: (-2.8595333, -2.3019171, -2.8595333, -2.3019171, -0.1667712, 0.1667712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.00 + 33.25 = 56.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0683536, upper bound: 0.0683538

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676402, upper bound: 0.0683532
time: 2.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683528, upper bound: 0.0683530
time: 5.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.02 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.02
Output dim: 7, lower bound: -0.0676402, upper bound: 0.0683532
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.02
Output dim: 7, lower bound: -0.0683528, upper bound: 0.0683530

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.8637915, -9.9785328, -10.8638773, -9.9781036, -0.2801673, 0.2797663
1: -6.5385680, -6.0593519, -6.5393810, -6.0593290, -0.2112714, 0.2120708
2: -8.3682871, -7.7011695, -8.3683205, -7.7007627, -0.2244664, 0.2240424
3: -2.2468474, -1.6026843, -2.2469296, -1.6022123, -0.2964585, 0.2959604
4: -7.7195673, -6.9018745, -7.7199454, -6.9018259, -0.2884115, 0.2888376
5: -8.0001822, -7.2465687, -8.0003052, -7.2461777, -0.4645348, 0.4640670
6: -13.4017286, -12.6410151, -13.4017735, -12.6402531, -0.3324931, 0.3317528
7: 5.5844350, 5.9946294, 5.5837002, 5.9946589, -0.1176624, 0.1184364
8: -2.0689640, -1.3921247, -2.0689893, -1.3892493, -0.2616911, 0.2588029
9: -2.8592353, -2.3019538, -2.8594494, -2.3019290, -0.1662984, 0.1665360

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 568

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0676402, upper bound: 0.0676404
time: 2.90 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676402, upper bound: 0.0683532
time: 2.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.8677979, -9.9778748, -10.8639107, -9.9779358, -0.2843280, 0.2822887
1: -6.5397654, -6.0528879, -6.5396981, -6.0593204, -0.2144542, 0.2176478
2: -8.3721819, -7.7005944, -8.3683338, -7.7006025, -0.2282779, 0.2266072
3: -2.2502303, -1.6017060, -2.2469616, -1.6020272, -0.3000093, 0.2980299
4: -7.7202816, -6.8991728, -7.7200947, -6.9018078, -0.2897415, 0.2917225
5: -8.0037937, -7.2456322, -8.0003510, -7.2460260, -0.4687057, 0.4660301
6: -13.4068890, -12.6396236, -13.4017925, -12.6399555, -0.3380287, 0.3327837
7: 5.5832181, 5.9995680, 5.5834112, 5.9946699, -0.1187927, 0.1214830
8: -2.0869322, -1.3880711, -2.0689993, -1.3881240, -0.2671772, 0.2621636
9: -2.8596897, -2.3001947, -2.8595333, -2.3019185, -0.1672307, 0.1683686

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 568

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683530, upper bound: 0.0676403
time: 3.00 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683530, upper bound: 0.0676403
time: 2.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.89 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 27.89
Output dim: 7, lower bound: -0.0676402, upper bound: 0.0676404
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 7, lower bound: -0.0676402, upper bound: 0.0683532
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 7, lower bound: -0.0683530, upper bound: 0.0676403
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 7, lower bound: -0.0683530, upper bound: 0.0676403

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.8637915, -9.9785328, -10.8677998, -9.9778786, -0.2802424, 0.2833908
1: -6.5385680, -6.0593519, -6.5397649, -6.0529461, -0.2161157, 0.2122265
2: -8.3682871, -7.7011695, -8.3721657, -7.7005920, -0.2243502, 0.2273167
3: -2.2468474, -1.6026843, -2.2502241, -1.6017073, -0.2965970, 0.2990496
4: -7.7195673, -6.9018745, -7.7202797, -6.8991728, -0.2910428, 0.2890040
5: -8.0001822, -7.2465687, -8.0037861, -7.2456322, -0.4649143, 0.4679370
6: -13.4017286, -12.6410151, -13.4068890, -12.6396379, -0.3329976, 0.3369472
7: 5.5844350, 5.9946294, 5.5832181, 5.9995670, -0.1203685, 0.1187626
8: -2.0689640, -1.3921247, -2.0869322, -1.3880739, -0.2628369, 0.2630856
9: -2.8592353, -2.3019538, -2.8596873, -2.3001947, -0.1679281, 0.1666088

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 58

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676380, upper bound: 0.0680181
time: 2.74 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0676400, upper bound: 0.0683529
time: 2.75 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.8677998, -9.9778786, -10.8637915, -9.9785328, -0.2833908, 0.2802423
1: -6.5397649, -6.0529461, -6.5385680, -6.0593519, -0.2122266, 0.2161157
2: -8.3721657, -7.7005920, -8.3682871, -7.7011695, -0.2273167, 0.2243502
3: -2.2502241, -1.6017073, -2.2468474, -1.6026843, -0.2990496, 0.2965968
4: -7.7202797, -6.8991728, -7.7195673, -6.9018745, -0.2890038, 0.2910428
5: -8.0037861, -7.2456322, -8.0001822, -7.2465687, -0.4679370, 0.4649143
6: -13.4068890, -12.6396379, -13.4017286, -12.6410151, -0.3369470, 0.3329976
7: 5.5832181, 5.9995670, 5.5844350, 5.9946294, -0.1187626, 0.1203685
8: -2.0869322, -1.3880739, -2.0689640, -1.3921247, -0.2630856, 0.2628369
9: -2.8596873, -2.3001947, -2.8592353, -2.3019538, -0.1666088, 0.1679282

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 58

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683505, upper bound: 0.0673053
time: 2.74 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683525, upper bound: 0.0676401
time: 2.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.8677979, -9.9778748, -10.8677979, -9.9778748, -0.2827365, 0.2827367
1: -6.5397654, -6.0528879, -6.5397654, -6.0528879, -0.2145422, 0.2145422
2: -8.3721819, -7.7005944, -8.3721819, -7.7005944, -0.2268591, 0.2268591
3: -2.2502303, -1.6017060, -2.2502303, -1.6017060, -0.2982132, 0.2982132
4: -7.7202816, -6.8991728, -7.7202816, -6.8991728, -0.2899873, 0.2899873
5: -8.0037937, -7.2456322, -8.0037937, -7.2456322, -0.4672999, 0.4672999
6: -13.4068890, -12.6396236, -13.4068890, -12.6396236, -0.3341539, 0.3341539
7: 5.5832181, 5.9995680, 5.5832181, 5.9995680, -0.1191845, 0.1191845
8: -2.0869322, -1.3880711, -2.0869322, -1.3880711, -0.2622685, 0.2622685
9: -2.8596897, -2.3001947, -2.8596897, -2.3001947, -0.1673846, 0.1673846

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 58

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683508, upper bound: 0.0673053
time: 2.97 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683528, upper bound: 0.0676399
time: 2.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 27.11 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.11
Output dim: 7, lower bound: -0.0676380, upper bound: 0.0680181
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.11
Output dim: 7, lower bound: -0.0676400, upper bound: 0.0683529
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.11
Output dim: 7, lower bound: -0.0683505, upper bound: 0.0673053
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.11
Output dim: 7, lower bound: -0.0683525, upper bound: 0.0676401
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.11
Output dim: 7, lower bound: -0.0683508, upper bound: 0.0673053
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.11
Output dim: 7, lower bound: -0.0683528, upper bound: 0.0676399

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.8637094, -9.9785328, -10.8677521, -9.9778786, -0.2801604, 0.2833440
1: -6.5385675, -6.0595455, -6.5397644, -6.0530577, -0.2159780, 0.2122235
2: -8.3681946, -7.7011681, -8.3721104, -7.7005920, -0.2243490, 0.2273159
3: -2.2468474, -1.6026959, -2.2502241, -1.6017135, -0.2965903, 0.2990386
4: -7.7195683, -6.9020243, -7.7202797, -6.8992586, -0.2909555, 0.2888533
5: -8.0001774, -7.2465687, -8.0037870, -7.2456322, -0.4648671, 0.4679098
6: -13.4017315, -12.6410131, -13.4068890, -12.6396389, -0.3329976, 0.3369467
7: 5.5844350, 5.9944954, 5.5832181, 5.9994888, -0.1202734, 0.1187603
8: -2.0687547, -1.3921251, -2.0868120, -1.3880739, -0.2628344, 0.2629358
9: -2.8588820, -2.3019528, -2.8594847, -2.3001947, -0.1675750, 0.1664052

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 58

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0680181
time: 3.68 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0680181
time: 3.86 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.8637886, -9.9784241, -10.8677979, -9.9778786, -0.2801843, 0.2835008
1: -6.5388284, -6.0593538, -6.5397644, -6.0529490, -0.2161157, 0.2122806
2: -8.3682871, -7.7010455, -8.3721647, -7.7005920, -0.2243763, 0.2271504
3: -2.2468622, -1.6026847, -2.2502241, -1.6017069, -0.2966118, 0.2990417
4: -7.7197666, -6.9018755, -7.7202792, -6.8991723, -0.2912440, 0.2888415
5: -8.0001202, -7.2466135, -8.0037527, -7.2456322, -0.4648533, 0.4679999
6: -13.4017048, -12.6410484, -13.4068928, -12.6396599, -0.3329508, 0.3369386
7: 5.5842547, 5.9946289, 5.5832181, 5.9995680, -0.1203684, 0.1188000
8: -2.0689602, -1.3918462, -2.0869308, -1.3880734, -0.2628964, 0.2630857
9: -2.8592343, -2.3014817, -2.8596869, -2.3001938, -0.1676790, 0.1670814

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 58

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0683503
time: 4.12 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0683530
time: 3.18 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.8677197, -9.9778786, -10.8637428, -9.9785328, -0.2833095, 0.2801951
1: -6.5397649, -6.0531411, -6.5385675, -6.0594635, -0.2122246, 0.2159196
2: -8.3720713, -7.7005930, -8.3682327, -7.7011695, -0.2273157, 0.2243495
3: -2.2502241, -1.6017182, -2.2468474, -1.6026903, -0.2990434, 0.2965858
4: -7.7202806, -6.8993225, -7.7195673, -6.9019613, -0.2889172, 0.2908916
5: -8.0037861, -7.2456322, -8.0001783, -7.2465687, -0.4678907, 0.4648867
6: -13.4068909, -12.6396389, -13.4017286, -12.6410122, -0.3369465, 0.3329973
7: 5.5832181, 5.9994330, 5.5844350, 5.9945526, -0.1187612, 0.1202331
8: -2.0867238, -1.3880758, -2.0688434, -1.3921247, -0.2628725, 0.2628355
9: -2.8593345, -2.3001947, -2.8590326, -2.3019543, -0.1662553, 0.1677247

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 58

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0673053
time: 2.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0673053
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.8677959, -9.9777699, -10.8637877, -9.9785328, -0.2833331, 0.2803516
1: -6.5400252, -6.0529480, -6.5385675, -6.0593514, -0.2122777, 0.2159783
2: -8.3721638, -7.7004685, -8.3682871, -7.7011709, -0.2273431, 0.2241838
3: -2.2502370, -1.6017066, -2.2468474, -1.6026838, -0.2990646, 0.2965889
4: -7.7204823, -6.8991728, -7.7195663, -6.9018755, -0.2892052, 0.2908802
5: -8.0037270, -7.2456770, -8.0001450, -7.2465687, -0.4678764, 0.4649763
6: -13.4068623, -12.6396742, -13.4017277, -12.6410332, -0.3368990, 0.3329895
7: 5.5830369, 5.9995675, 5.5844350, 5.9946299, -0.1185731, 0.1202734
8: -2.0869274, -1.3877950, -2.0689611, -1.3921242, -0.2629359, 0.2624621
9: -2.8596864, -2.2997208, -2.8592343, -2.3019533, -0.1663593, 0.1684008

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 58

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0676382
time: 2.77 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0676382
time: 2.79 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.8677197, -9.9778748, -10.8677530, -9.9778748, -0.2826550, 0.2826897
1: -6.5397649, -6.0530825, -6.5397654, -6.0530005, -0.2145406, 0.2145395
2: -8.3720894, -7.7005944, -8.3721285, -7.7005935, -0.2268580, 0.2268581
3: -2.2502303, -1.6017182, -2.2502303, -1.6017122, -0.2982066, 0.2982020
4: -7.7202787, -6.8993225, -7.7202816, -6.8992586, -0.2899008, 0.2898366
5: -8.0037918, -7.2456322, -8.0037918, -7.2456322, -0.4672523, 0.4672728
6: -13.4068890, -12.6396227, -13.4068890, -12.6396208, -0.3341537, 0.3341537
7: 5.5832181, 5.9994345, 5.5832181, 5.9994903, -0.1191832, 0.1191823
8: -2.0867238, -1.3880720, -2.0868120, -1.3880711, -0.2622659, 0.2622669
9: -2.8593369, -2.3001947, -2.8594861, -2.3001947, -0.1670313, 0.1671810

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 58

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680177, upper bound: 0.0673057
time: 2.90 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680177, upper bound: 0.0673053
time: 3.95 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.8677979, -9.9777660, -10.8677998, -9.9778748, -0.2826784, 0.2828462
1: -6.5400257, -6.0528898, -6.5397654, -6.0528889, -0.2145934, 0.2145969
2: -8.3721809, -7.7004685, -8.3721819, -7.7005920, -0.2268853, 0.2266924
3: -2.2502451, -1.6017076, -2.2502303, -1.6017077, -0.2982278, 0.2982047
4: -7.7204819, -6.8991728, -7.7202802, -6.8991723, -0.2901891, 0.2898256
5: -8.0037308, -7.2456770, -8.0037546, -7.2456322, -0.4672389, 0.4673624
6: -13.4068613, -12.6396570, -13.4068890, -12.6396437, -0.3341064, 0.3341451
7: 5.5830369, 5.9995689, 5.5832181, 5.9995675, -0.1189953, 0.1192219
8: -2.0869274, -1.3877926, -2.0869308, -1.3880706, -0.2623281, 0.2618937
9: -2.8596888, -2.2997208, -2.8596888, -2.3001947, -0.1671353, 0.1678571

Time for backsubstitution: 20.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 58

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680180, upper bound: 0.0676386
time: 2.79 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680177, upper bound: 0.0676403
time: 3.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.41 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0680181
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0680181
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0683503
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0673051, upper bound: 0.0683530
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0673053
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0673053
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0676382
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680176, upper bound: 0.0676382
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680177, upper bound: 0.0673057
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680177, upper bound: 0.0673053
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680180, upper bound: 0.0676386
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.41
Output dim: 7, lower bound: -0.0680177, upper bound: 0.0676403

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.8637094, -9.9785328, -10.8677197, -9.9778786, -0.2801602, 0.2833093
1: -6.5385675, -6.0595455, -6.5397649, -6.0531411, -0.2159197, 0.2122234
2: -8.3681946, -7.7011681, -8.3720713, -7.7005930, -0.2243491, 0.2273158
3: -2.2468474, -1.6026959, -2.2502241, -1.6017182, -0.2965858, 0.2990386
4: -7.7195683, -6.9020243, -7.7202806, -6.8993225, -0.2908916, 0.2888532
5: -8.0001774, -7.2465687, -8.0037861, -7.2456322, -0.4648671, 0.4678907
6: -13.4017315, -12.6410131, -13.4068909, -12.6396389, -0.3329978, 0.3369462
7: 5.5844350, 5.9944954, 5.5832181, 5.9994330, -0.1202331, 0.1187603
8: -2.0687547, -1.3921251, -2.0867238, -1.3880758, -0.2628344, 0.2628725
9: -2.8588820, -2.3019528, -2.8593345, -2.3001947, -0.1675749, 0.1662552

Time for backsubstitution: 20.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675700, upper bound: 0.0680146
time: 4.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675919, upper bound: 0.0680122
time: 3.12 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.8637094, -9.9785328, -10.8677959, -9.9777699, -0.2802701, 0.2833908
1: -6.5385675, -6.0595455, -6.5400252, -6.0529480, -0.2159780, 0.2122879
2: -8.3681946, -7.7011681, -8.3721638, -7.7004685, -0.2242577, 0.2271920
3: -2.2468474, -1.6026959, -2.2502370, -1.6017066, -0.2965968, 0.2990537
4: -7.7195683, -6.9020243, -7.7204823, -6.8991728, -0.2910416, 0.2890552
5: -8.0001774, -7.2465687, -8.0037270, -7.2456770, -0.4649301, 0.4679365
6: -13.4017315, -12.6410131, -13.4068623, -12.6396742, -0.3329625, 0.3369200
7: 5.5844350, 5.9944954, 5.5830369, 5.9995675, -0.1202734, 0.1186613
8: -2.0687547, -1.3921251, -2.0869274, -1.3877950, -0.2626284, 0.2629359
9: -2.8588820, -2.3019528, -2.8596864, -2.2997208, -0.1680484, 0.1666071

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675700, upper bound: 0.0680146
time: 3.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675919, upper bound: 0.0680120
time: 4.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.8637886, -9.9784241, -10.8677197, -9.9778786, -0.2802417, 0.2834191
1: -6.5388284, -6.0593538, -6.5397649, -6.0531411, -0.2159196, 0.2122113
2: -8.3682871, -7.7010455, -8.3720713, -7.7005930, -0.2242253, 0.2272245
3: -2.2468622, -1.6026847, -2.2502241, -1.6017182, -0.2966006, 0.2990499
4: -7.7197666, -6.9018755, -7.7202806, -6.8993225, -0.2910936, 0.2890029
5: -8.0001202, -7.2466135, -8.0037861, -7.2456322, -0.4649134, 0.4679537
6: -13.4017048, -12.6410484, -13.4068909, -12.6396389, -0.3329720, 0.3369112
7: 5.5842547, 5.9946289, 5.5832181, 5.9994330, -0.1202331, 0.1186712
8: -2.0689602, -1.3918462, -2.0867238, -1.3880758, -0.2625555, 0.2628725
9: -2.8592343, -2.3014817, -2.8593345, -2.3001947, -0.1679268, 0.1667288

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672776, upper bound: 0.0683474
time: 3.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672995, upper bound: 0.0683450
time: 4.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.8637886, -9.9784241, -10.8677959, -9.9777699, -0.2801843, 0.2833327
1: -6.5388284, -6.0593538, -6.5400252, -6.0529480, -0.2161158, 0.2122806
2: -8.3682871, -7.7010455, -8.3721638, -7.7004685, -0.2243763, 0.2273431
3: -2.2468622, -1.6026847, -2.2502370, -1.6017066, -0.2965887, 0.2990417
4: -7.7197666, -6.9018755, -7.7204823, -6.8991728, -0.2908800, 0.2888416
5: -8.0001202, -7.2466135, -8.0037270, -7.2456770, -0.4648542, 0.4678764
6: -13.4017048, -12.6410484, -13.4068623, -12.6396742, -0.3329897, 0.3369384
7: 5.5842547, 5.9946289, 5.5830369, 5.9995675, -0.1202734, 0.1187999
8: -2.0689602, -1.3918462, -2.0869274, -1.3877950, -0.2628964, 0.2629468
9: -2.8592343, -2.3014817, -2.8596864, -2.2997208, -0.1676793, 0.1663594

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672776, upper bound: 0.0683489
time: 3.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0672995, upper bound: 0.0683475
time: 2.99 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.8677197, -9.9778786, -10.8637094, -9.9785328, -0.2833092, 0.2801603
1: -6.5397649, -6.0531411, -6.5385675, -6.0595455, -0.2122235, 0.2159196
2: -8.3720713, -7.7005930, -8.3681946, -7.7011681, -0.2273158, 0.2243491
3: -2.2502241, -1.6017182, -2.2468474, -1.6026959, -0.2990386, 0.2965858
4: -7.7202806, -6.8993225, -7.7195683, -6.9020243, -0.2888532, 0.2908916
5: -8.0037861, -7.2456322, -8.0001774, -7.2465687, -0.4678907, 0.4648671
6: -13.4068909, -12.6396389, -13.4017315, -12.6410131, -0.3369462, 0.3329978
7: 5.5832181, 5.9994330, 5.5844350, 5.9944954, -0.1187603, 0.1202331
8: -2.0867238, -1.3880758, -2.0687547, -1.3921251, -0.2628725, 0.2628345
9: -2.8593345, -2.3001947, -2.8588820, -2.3019528, -0.1662552, 0.1675748

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682825, upper bound: 0.0673018
time: 2.83 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683044, upper bound: 0.0672997
time: 2.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -10.8677197, -9.9778786, -10.8637886, -9.9784241, -0.2834191, 0.2802418
1: -6.5397649, -6.0531411, -6.5388284, -6.0593538, -0.2122113, 0.2159196
2: -8.3720713, -7.7005930, -8.3682871, -7.7010455, -0.2272245, 0.2242253
3: -2.2502241, -1.6017182, -2.2468622, -1.6026847, -0.2990499, 0.2966006
4: -7.7202806, -6.8993225, -7.7197666, -6.9018755, -0.2890029, 0.2910936
5: -8.0037861, -7.2456322, -8.0001202, -7.2466135, -0.4679537, 0.4649134
6: -13.4068909, -12.6396389, -13.4017048, -12.6410484, -0.3369112, 0.3329720
7: 5.5832181, 5.9994330, 5.5842547, 5.9946289, -0.1186712, 0.1202331
8: -2.0867238, -1.3880758, -2.0689602, -1.3918462, -0.2628725, 0.2625554
9: -2.8593345, -2.3001947, -2.8592343, -2.3014817, -0.1667288, 0.1679268

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0682825, upper bound: 0.0673018
time: 2.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0683044, upper bound: 0.0672997
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.8677959, -9.9777699, -10.8637094, -9.9785328, -0.2833908, 0.2802700
1: -6.5400252, -6.0529480, -6.5385675, -6.0595455, -0.2122879, 0.2159779
2: -8.3721638, -7.7004685, -8.3681946, -7.7011681, -0.2271920, 0.2242577
3: -2.2502370, -1.6017066, -2.2468474, -1.6026959, -0.2990537, 0.2965968
4: -7.7204823, -6.8991728, -7.7195683, -6.9020243, -0.2890553, 0.2910416
5: -8.0037270, -7.2456770, -8.0001774, -7.2465687, -0.4679365, 0.4649301
6: -13.4068623, -12.6396742, -13.4017315, -12.6410131, -0.3369200, 0.3329625
7: 5.5830369, 5.9995675, 5.5844350, 5.9944954, -0.1186613, 0.1202734
8: -2.0869274, -1.3877950, -2.0687547, -1.3921251, -0.2629359, 0.2626283
9: -2.8596864, -2.2997208, -2.8588820, -2.3019528, -0.1666071, 0.1680484

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679901, upper bound: 0.0676346
time: 2.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680120, upper bound: 0.0676325
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.8677959, -9.9777699, -10.8637886, -9.9784241, -0.2833328, 0.2801844
1: -6.5400252, -6.0529480, -6.5388284, -6.0593538, -0.2122806, 0.2159780
2: -8.3721638, -7.7004685, -8.3682871, -7.7010455, -0.2273431, 0.2243763
3: -2.2502370, -1.6017066, -2.2468622, -1.6026847, -0.2990417, 0.2965887
4: -7.7204823, -6.8991728, -7.7197666, -6.9018755, -0.2888416, 0.2908801
5: -8.0037270, -7.2456770, -8.0001202, -7.2466135, -0.4678764, 0.4648542
6: -13.4068623, -12.6396742, -13.4017048, -12.6410484, -0.3369384, 0.3329897
7: 5.5830369, 5.9995675, 5.5842547, 5.9946289, -0.1187999, 0.1202734
8: -2.0869274, -1.3877950, -2.0689602, -1.3918462, -0.2629359, 0.2628963
9: -2.8596864, -2.2997208, -2.8592343, -2.3014817, -0.1663594, 0.1676792

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679901, upper bound: 0.0676347
time: 2.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0680120, upper bound: 0.0676347
time: 2.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.8677197, -9.9778748, -10.8677197, -9.9778748, -0.2826548, 0.2826549
1: -6.5397649, -6.0530825, -6.5397649, -6.0530825, -0.2145395, 0.2145395
2: -8.3720894, -7.7005944, -8.3720894, -7.7005944, -0.2268578, 0.2268579
3: -2.2502303, -1.6017182, -2.2502303, -1.6017182, -0.2982018, 0.2982018
4: -7.7202787, -6.8993225, -7.7202787, -6.8993225, -0.2898365, 0.2898366
5: -8.0037918, -7.2456322, -8.0037918, -7.2456322, -0.4672527, 0.4672527
6: -13.4068890, -12.6396227, -13.4068890, -12.6396227, -0.3341534, 0.3341534
7: 5.5832181, 5.9994345, 5.5832181, 5.9994345, -0.1191822, 0.1191823
8: -2.0867238, -1.3880720, -2.0867238, -1.3880720, -0.2622660, 0.2622662
9: -2.8593369, -2.3001947, -2.8593369, -2.3001947, -0.1670312, 0.1670312

Time for backsubstitution: 22.49 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.25 + 563.80 = 620.05 seconds
