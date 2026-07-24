## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 147.6105270206


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288)
1: (-23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952)
2: (-12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484)
3: (-17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907)
4: (-24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.00 + 1.87 = 3.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -147.9063397, upper bound: 147.9063397

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9059183, upper bound: 147.9050160
time: 0.86 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9061533, upper bound: 147.9061533
time: 0.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -147.9059183, upper bound: 147.9050160
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -147.9061533, upper bound: 147.9061533

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -32.2205009, 114.6340027, -35.5556183, 127.1783447, -159.3988495, 150.1896057
1: -20.1563721, 69.6905975, -22.3797932, 77.1609421, -97.3173065, 92.0703812
2: -11.1342382, 64.6120987, -12.3634720, 71.4373093, -82.5715485, 76.9755707
3: -15.5158892, 95.4042130, -17.1787357, 105.7507629, -121.2666550, 112.5829468
4: -21.1636353, 78.9279785, -23.5327110, 87.3604889, -108.5241241, 102.4606857

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9021434
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9026422, upper bound: 147.9028306
time: 0.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -52.8363609, 195.9806824, -35.7382126, 128.0534363, -180.8898010, 231.7189026
1: -34.2473640, 114.4997253, -22.4836864, 77.5905380, -111.8378983, 136.9834137
2: -18.7401466, 104.8504105, -12.4142675, 71.8288574, -90.5689926, 117.2646713
3: -25.9337959, 158.7232056, -17.2549076, 106.3378754, -132.2716675, 175.9781189
4: -35.6399002, 129.6212158, -23.6402225, 87.8260727, -123.4659576, 153.2614441

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9025973, upper bound: 147.9038913
time: 0.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9054673, upper bound: 147.9054673
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.46 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9021434
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -147.9026422, upper bound: 147.9028306
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -147.9025973, upper bound: 147.9038913
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -147.9054673, upper bound: 147.9054673

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -27.4858093, 97.6869736, -22.7208843, 80.5440826, -108.0298843, 120.4078522
1: -17.1199856, 58.5421486, -14.0927296, 47.4779816, -64.5979614, 72.6348801
2: -9.4290838, 54.0503769, -7.8340440, 43.2499161, -52.6789970, 61.8844185
3: -13.2284641, 80.3810730, -10.9488325, 65.7518005, -78.9802628, 91.3299026
4: -17.9067497, 66.2671432, -14.8645678, 53.4381905, -71.3449402, 81.1316986

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9001723, upper bound: 147.9020140
time: 0.81 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -31.2575722, 111.0437088, -32.3822212, 115.4093781, -146.6669464, 143.4259338
1: -19.5012016, 67.5361862, -20.2685051, 70.1169434, -89.6181488, 87.8046875
2: -10.7550192, 62.6496201, -11.1589670, 64.9743195, -75.7293320, 73.8085861
3: -15.0153341, 92.4176636, -15.5709381, 96.0126190, -111.0279541, 107.9886017
4: -20.4486504, 76.4875717, -21.2422142, 79.3618851, -99.8105316, 97.7297821

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9026422, upper bound: 147.9026975
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9026422, upper bound: 147.9028306
time: 0.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -50.6986122, 188.0899200, -23.2786617, 82.8126526, -133.5112610, 211.3685760
1: -32.8817368, 109.6052246, -14.4929199, 48.7789726, -81.6606827, 124.0981445
2: -17.9804344, 100.2486115, -8.0530014, 44.4322319, -62.4126663, 108.3016129
3: -24.8946381, 152.0971375, -11.2386570, 67.5750961, -92.4697342, 163.3358002
4: -34.1707077, 124.0411911, -15.2888241, 54.9199295, -89.0906296, 139.3300171

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9019824
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9038913
time: 0.63 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -50.8521194, 188.8904114, -32.4856529, 115.9875793, -166.8396912, 221.3760681
1: -32.9876137, 109.8890533, -20.3285198, 70.3461838, -103.3337860, 130.2175751
2: -18.0127316, 100.4926682, -11.1886091, 65.1649628, -83.1776657, 111.6812744
3: -24.9519997, 152.5070953, -15.6180382, 96.3456955, -121.2976837, 168.1251068
4: -34.2615318, 124.3470230, -21.3051262, 79.6029739, -113.8645020, 145.6521454

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9038913, upper bound: 147.9025973
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9038913, upper bound: 147.9054673
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9001723, upper bound: 147.9020140
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9026422, upper bound: 147.9026975
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9026422, upper bound: 147.9028306
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9019824
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9038913
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9038913, upper bound: 147.9025973
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -147.9038913, upper bound: 147.9054673

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -25.2360840, 89.6257782, -22.2821808, 78.9040680, -104.1401520, 111.9079590
1: -15.6211863, 53.3701515, -13.7871780, 46.4668961, -62.0880814, 67.1573181
2: -8.5887718, 49.2353592, -7.6658044, 42.3204651, -50.9092369, 56.9011650
3: -12.1277075, 73.2938080, -10.7274132, 64.3464355, -76.4741440, 84.0212250
4: -16.3059845, 60.3937912, -14.5419626, 52.2914886, -68.5974731, 74.9357529

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020047
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020047
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -25.4645882, 90.9524155, -22.4295788, 79.5631943, -105.0277786, 113.3819733
1: -15.8575830, 54.1389999, -13.9209700, 46.8952789, -62.7528610, 68.0599670
2: -8.7423306, 49.8725281, -7.7416148, 42.7085342, -51.4508667, 57.6141434
3: -12.3463306, 74.4235916, -10.8255444, 64.9550171, -77.3013382, 85.2491302
4: -16.6409855, 61.2556305, -14.6891146, 52.7807884, -69.4217758, 75.9447479

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -19.5481339, 68.5877380, -32.3822212, 115.4093781, -134.9575043, 100.9699554
1: -12.0210724, 40.4643402, -20.2685051, 70.1169434, -82.1380157, 60.7328453
2: -6.6972446, 36.8099442, -11.1589670, 64.9743195, -71.6715622, 47.9689064
3: -9.4400902, 56.0563850, -15.5709381, 96.0126190, -105.4527130, 71.6273193
4: -12.6868353, 45.5176544, -21.2422142, 79.3618851, -92.0487137, 66.7598572

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9008172
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9026975
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -29.1594334, 103.3192902, -32.3822212, 115.4093781, -144.5688019, 135.7015076
1: -18.0970116, 62.8190041, -20.2685051, 70.1169434, -88.2139587, 83.0875092
2: -9.9481611, 58.3167686, -11.1589670, 64.9743195, -74.9224777, 69.4757309
3: -13.9435673, 85.8972778, -15.5709381, 96.0126190, -109.9561844, 101.4682159
4: -18.9120045, 71.1313782, -21.2422142, 79.3618851, -98.2738876, 92.3735809

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9008825
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9023109
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -38.7809143, 142.6156616, -23.2786617, 82.8126526, -121.5935669, 165.8943176
1: -24.6673088, 82.6511230, -14.4929199, 48.7789726, -73.4462509, 97.1440430
2: -13.4502430, 75.8942032, -8.0530014, 44.4322319, -57.8824730, 83.9472046
3: -18.8772545, 114.3800125, -11.2386570, 67.5750961, -86.4523392, 125.6186600
4: -25.4005585, 93.7441101, -15.2888241, 54.9199295, -80.3204727, 109.0329361

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018244, upper bound: 147.9018714
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018686, upper bound: 147.9018824
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -47.1223869, 175.4886322, -23.2786617, 82.8126526, -129.9350433, 198.7672882
1: -30.6360245, 101.5087738, -14.4929199, 48.7789726, -79.4149628, 116.0016937
2: -16.6670303, 92.6635742, -8.0530014, 44.4322319, -61.0992546, 100.7165756
3: -23.1523438, 141.1236877, -11.2386570, 67.5750961, -90.7274399, 152.3623505
4: -31.7165565, 114.8225174, -15.2888241, 54.9199295, -86.6364746, 130.1113434

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018244, upper bound: 147.9037642
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018686, upper bound: 147.9037413
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -38.7809143, 142.6156616, -32.4856529, 115.9875793, -154.7684784, 175.1013184
1: -24.6673088, 82.6511230, -20.3285198, 70.3461838, -95.0134888, 102.9796448
2: -13.4502430, 75.8942032, -11.1886091, 65.1649628, -78.6152039, 87.0828094
3: -18.8772545, 114.3800125, -15.6180382, 96.3456955, -115.2229385, 129.9980011
4: -25.4005585, 93.7441101, -21.3051262, 79.6029739, -105.0035095, 115.0492172

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9006125
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9013680
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.1223869, 175.4886322, -32.4856529, 115.9875793, -163.1099548, 207.9742889
1: -30.6360245, 101.5087738, -20.3285198, 70.3461838, -100.9822006, 121.8372955
2: -16.6670303, 92.6635742, -11.1886091, 65.1649628, -81.8319855, 103.8521805
3: -23.1523438, 141.1236877, -15.6180382, 96.3456955, -119.4980316, 156.7416992
4: -31.7165565, 114.8225174, -21.3051262, 79.6029739, -111.3195267, 136.1276398

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9026422
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9053290
time: 1.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.82 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020047
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020047
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9008172
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9026975
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9008825
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9006125, upper bound: 147.9023109
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9018244, upper bound: 147.9018714
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9018686, upper bound: 147.9018824
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9018244, upper bound: 147.9037642
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9018686, upper bound: 147.9037413
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9006125
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9013680
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9026422
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 0, lower bound: -147.9019824, upper bound: 147.9053290

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -25.2360840, 89.6257782, -20.0677776, 70.9013443, -96.1374283, 109.6935577
1: -15.6211863, 53.3701515, -12.2504187, 41.2731743, -56.8943596, 65.6205597
2: -8.5887718, 49.2353592, -6.8060398, 37.5283012, -46.1170731, 56.0413971
3: -12.1277075, 73.2938080, -9.6141233, 57.1452942, -69.2729950, 82.9079285
4: -16.3059845, 60.3937912, -12.8958855, 46.4018631, -62.7078476, 73.2896652

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019887
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -25.2360840, 89.6257782, -22.1974831, 79.1504288, -104.3865051, 111.8232574
1: -15.6211863, 53.3701515, -13.8312082, 46.3539200, -61.9751053, 67.2013474
2: -8.5887718, 49.2353592, -7.6847644, 42.1767960, -50.7655640, 56.9201241
3: -12.1277075, 73.2938080, -10.7949877, 64.3056946, -76.4334030, 84.0887833
4: -16.3059845, 60.3937912, -14.5783787, 52.2100220, -68.5160065, 74.9721680

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019887
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.4645882, 90.9524155, -20.0677776, 70.9013443, -96.3659363, 111.0201950
1: -15.8575830, 54.1389999, -12.2504187, 41.2731743, -57.1307564, 66.3894196
2: -8.7423306, 49.8725281, -6.8060398, 37.5283012, -46.2706299, 56.6785622
3: -12.3463306, 74.4235916, -9.6141233, 57.1452942, -69.4916153, 84.0377121
4: -16.6409855, 61.2556305, -12.8958855, 46.4018631, -63.0428467, 74.1515198

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.4645882, 90.9524155, -22.2335720, 79.2846985, -104.7492828, 113.1859818
1: -15.8575830, 54.1389999, -13.8561430, 46.4380569, -62.2956390, 67.9951401
2: -8.7423306, 49.8725281, -7.6999674, 42.2512589, -50.9935913, 57.5724869
3: -12.3463306, 74.4235916, -10.8138218, 64.4220734, -76.7684021, 85.2374115
4: -16.6409855, 61.2556305, -14.6088352, 52.3037872, -68.9447708, 75.8644638

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -19.5481339, 68.5877380, -29.1594334, 103.3192902, -122.8674240, 97.7471695
1: -12.0210724, 40.4643402, -18.0970116, 62.8190041, -74.8400726, 58.5613518
2: -6.6972446, 36.8099442, -9.9481611, 58.3167686, -65.0140076, 46.7581062
3: -9.4400902, 56.0563850, -13.9435673, 85.8972778, -95.3373718, 69.9999542
4: -12.6868353, 45.5176544, -18.9120045, 71.1313782, -83.8181992, 64.4296570

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020862, upper bound: 147.9005476
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020195, upper bound: 147.9005476
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -19.5481339, 68.5877380, -41.9033279, 154.8003845, -174.3485107, 110.4910507
1: -12.0210724, 40.4643402, -27.1309776, 90.2835007, -102.3045731, 67.5953217
2: -6.6972446, 36.8099442, -14.7718201, 82.5081940, -89.2054367, 51.5817642
3: -9.4400902, 56.0563850, -20.5379486, 125.2913132, -134.7313843, 76.5943298
4: -12.6868353, 45.5176544, -28.1695213, 102.0735474, -114.7603760, 73.6871796

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020862, upper bound: 147.9025467
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018824, upper bound: 147.9025399
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -29.1594334, 103.3192902, -29.1594334, 103.3192902, -132.4787140, 132.4787140
1: -18.0970116, 62.8190041, -18.0970116, 62.8190041, -80.9160156, 80.9160156
2: -9.9481611, 58.3167686, -9.9481611, 58.3167686, -68.2649231, 68.2649231
3: -13.9435673, 85.8972778, -13.9435673, 85.8972778, -99.8408432, 99.8408432
4: -18.9120045, 71.1313782, -18.9120045, 71.1313782, -90.0433731, 90.0433731

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8606232, upper bound: 147.8758662
time: 2.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9005970, upper bound: 147.9008247
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -29.1594334, 103.3192902, -45.2306366, 167.9722137, -197.1316528, 148.5499268
1: -18.0970116, 62.8190041, -29.3667812, 97.4092255, -115.5062408, 92.1857834
2: -9.9481611, 58.3167686, -15.9815493, 88.9470825, -98.8952408, 74.2983017
3: -13.9435673, 85.8972778, -22.2075253, 135.3625793, -149.3061523, 108.1048050
4: -18.9120045, 71.1313782, -30.4302330, 110.1680145, -129.0800171, 101.5615845

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8606232, upper bound: 147.8764277
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9005970, upper bound: 147.9023109
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -38.5543289, 141.8106384, -20.5580711, 72.8395386, -111.3938675, 162.3687134
1: -24.5124798, 82.1509705, -12.6048326, 42.4445572, -66.9570389, 94.7557907
2: -13.3628445, 75.4435806, -7.0036321, 38.5909653, -51.9538040, 82.4472046
3: -18.7613163, 113.6783066, -9.8690500, 58.7866478, -77.5479279, 123.5473557
4: -25.2316990, 93.1794891, -13.2769909, 47.7300758, -72.9617538, 106.4564819

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018249
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018575
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -37.9256554, 139.6205292, -22.4526672, 80.2389984, -118.1646423, 162.0731964
1: -24.1211815, 80.7622147, -14.0240250, 46.9927139, -71.1138916, 94.7862320
2: -13.1473007, 74.1491547, -7.7921991, 42.7557449, -55.9030457, 81.9413452
3: -18.4705448, 111.7863541, -10.9185658, 65.2038727, -83.6744156, 122.7049026
4: -24.8187523, 91.6088028, -14.7857008, 52.9388504, -77.7575989, 106.3945007

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018575, upper bound: 147.9018248
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018575, upper bound: 147.9018824
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.8990211, 174.6907501, -20.5580711, 72.8395386, -119.7385559, 195.2488251
1: -30.4879189, 100.9997101, -12.6048326, 42.4445572, -72.9324799, 113.6045380
2: -16.5829163, 92.1960678, -7.0036321, 38.5909653, -55.1738815, 99.1996994
3: -23.0385094, 140.4252625, -9.8690500, 58.7866478, -81.8251266, 150.2943115
4: -31.5534439, 114.2476959, -13.2769909, 47.7300758, -79.2835007, 127.5246811

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8963712, upper bound: 147.8979023
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9036074
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.0344353, 171.6449738, -22.4526672, 80.2389984, -126.2734375, 194.0976410
1: -29.9337845, 99.1241913, -14.0240250, 46.9927139, -76.9264984, 113.1482086
2: -16.2762184, 90.4710541, -7.7921991, 42.7557449, -59.0319633, 98.2632446
3: -22.6276932, 137.8368073, -10.9185658, 65.2038727, -87.8315353, 148.7553711
4: -30.9657612, 112.1220551, -14.7857008, 52.9388504, -83.9045944, 126.9077377

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8964593, upper bound: 147.8980048
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9035858
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -38.7809143, 142.6156616, -29.1354561, 103.2347565, -142.0156403, 171.7510986
1: -24.6673088, 82.6511230, -18.0814457, 62.7586708, -87.4259796, 100.7325668
2: -13.4502430, 75.8942032, -9.9390821, 58.2590294, -71.7092667, 85.8332825
3: -18.8772545, 114.3800125, -13.9320250, 85.8168640, -104.6941147, 128.3119812
4: -25.4005585, 93.7441101, -18.8949451, 71.0639191, -96.4644699, 112.6390533

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020355, upper bound: 147.8999914
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020140, upper bound: 147.9001723
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -38.7809143, 142.6156616, -47.1223869, 175.4886322, -214.0293274, 189.6379242
1: -24.6673088, 82.6511230, -30.6360245, 101.5087738, -126.1374817, 113.2871475
2: -13.4502430, 75.8942032, -16.6670303, 92.6635742, -106.1138153, 92.5612335
3: -18.8772545, 114.3800125, -23.1523438, 141.1236877, -160.0009308, 137.5323181
4: -25.4005585, 93.7441101, -31.7165565, 114.8225174, -140.2230835, 125.4606628

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020355, upper bound: 147.9009292
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018824, upper bound: 147.9011945
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -45.2306366, 167.9722137, -29.1354561, 103.2347565, -148.4653931, 197.1076508
1: -29.3667812, 97.4092255, -18.0814457, 62.7586708, -92.1254501, 115.4906693
2: -15.9815493, 88.9470825, -9.9390821, 58.2590294, -74.2405624, 98.8861618
3: -22.2075253, 135.3625793, -13.9320250, 85.8168640, -108.0243759, 149.2945862
4: -30.4302330, 110.1680145, -18.8949451, 71.0639191, -101.4941406, 129.0629578

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8867563, upper bound: 147.8954353
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9025690, upper bound: 147.9022985
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.1223869, 175.4886322, -47.1223869, 175.4886322, -222.3931274, 222.3931274
1: -30.6360245, 101.5087738, -30.6360245, 101.5087738, -132.0459747, 132.0459747
2: -16.6670303, 92.6635742, -16.6670303, 92.6635742, -109.3306046, 109.3306046
3: -23.1523438, 141.1236877, -23.1523438, 141.1236877, -164.2760315, 164.2760315
4: -31.7165565, 114.8225174, -31.7165565, 114.8225174, -146.5390778, 146.5390778

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8867563, upper bound: 147.8954353
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9025690, upper bound: 147.9051910
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.46 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019887
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019887
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9020862, upper bound: 147.9005476
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9020195, upper bound: 147.9005476
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9020862, upper bound: 147.9025467
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9018824, upper bound: 147.9025399
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8606232, upper bound: 147.8758662
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9005970, upper bound: 147.9008247
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8606232, upper bound: 147.8764277
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9005970, upper bound: 147.9023109
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018249
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018575
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9018575, upper bound: 147.9018248
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9018575, upper bound: 147.9018824
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8963712, upper bound: 147.8979023
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9036074
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8964593, upper bound: 147.8980048
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9035858
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9020355, upper bound: 147.8999914
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9020140, upper bound: 147.9001723
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9020355, upper bound: 147.9009292
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9018824, upper bound: 147.9011945
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8867563, upper bound: 147.8954353
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9025690, upper bound: 147.9022985
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.8867563, upper bound: 147.8954353
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 0, lower bound: -147.9025690, upper bound: 147.9051910

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -20.0677776, 70.9013443, -88.0736618, 80.1966019
1: -10.3972998, 34.8535805, -12.2504187, 41.2731743, -51.6704712, 47.1040001
2: -5.7887311, 31.5996094, -6.8060398, 37.5283012, -43.3170319, 38.4056473
3: -8.2676754, 48.3058929, -9.6141233, 57.1452942, -65.4129715, 57.9200134
4: -10.9508162, 39.1615753, -12.8958855, 46.4018631, -57.3526764, 52.0574608

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020029
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020029
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -26.5009613, 93.8624191, -20.0677776, 70.9013443, -97.4022980, 113.9301987
1: -16.3315239, 56.8662643, -12.2504187, 41.2731743, -57.6046944, 69.1166763
2: -8.9596434, 52.8075600, -6.8060398, 37.5283012, -46.4879379, 59.6135979
3: -12.6496582, 77.6712112, -9.6141233, 57.1452942, -69.7949524, 87.2853241
4: -17.0362015, 64.3830643, -12.8958855, 46.4018631, -63.4380646, 77.2789459

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020350
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020355
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -22.1974831, 79.1504288, -96.3227463, 82.3263016
1: -10.3972998, 34.8535805, -13.8312082, 46.3539200, -56.7512207, 48.6847878
2: -5.7887311, 31.5996094, -7.6847644, 42.1767960, -47.9655228, 39.2843742
3: -8.2676754, 48.3058929, -10.7949877, 64.3056946, -72.5733719, 59.1008797
4: -10.9508162, 39.1615753, -14.5783787, 52.2100220, -63.1608353, 53.7399521

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019852
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019887
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.5009613, 93.8624191, -22.1974831, 79.1504288, -105.6513824, 116.0598984
1: -16.3315239, 56.8662643, -13.8312082, 46.3539200, -62.6854401, 70.6974716
2: -8.9596434, 52.8075600, -7.6847644, 42.1767960, -51.1364288, 60.4923248
3: -12.6496582, 77.6712112, -10.7949877, 64.3056946, -76.9553528, 88.4661789
4: -17.0362015, 64.3830643, -14.5783787, 52.2100220, -69.2462234, 78.9614410

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019973
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.2246304, 67.9323654, -20.0677776, 70.9013443, -90.1259766, 88.0001450
1: -11.8859072, 39.8227730, -12.2504187, 41.2731743, -53.1590767, 52.0731888
2: -6.6206121, 36.1757812, -6.8060398, 37.5283012, -44.1489105, 42.9818115
3: -9.4038820, 55.2570419, -9.6141233, 57.1452942, -66.5491714, 64.8711624
4: -12.5422478, 44.8210831, -12.8958855, 46.4018631, -58.9441071, 57.7169685

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020029
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020029
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.7230053, 91.4297180, -20.0677776, 70.9013443, -96.6243515, 111.4974976
1: -15.8495493, 55.2890701, -12.2504187, 41.2731743, -57.1227226, 67.5394669
2: -8.6752014, 51.3622665, -6.8060398, 37.5283012, -46.2034988, 58.1683006
3: -12.3136873, 75.5141220, -9.6141233, 57.1452942, -69.4589767, 85.1282349
4: -16.5161533, 62.5906906, -12.8958855, 46.4018631, -62.9180145, 75.4865723

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020350
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020355
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19.2246304, 67.9323654, -22.2335720, 79.2846985, -98.5093307, 90.1659393
1: -11.8859072, 39.8227730, -13.8561430, 46.4380569, -58.3239632, 53.6789131
2: -6.6206121, 36.1757812, -7.6999674, 42.2512589, -48.8718643, 43.8757362
3: -9.4038820, 55.2570419, -10.8138218, 64.4220734, -73.8259583, 66.0708618
4: -12.5422478, 44.8210831, -14.6088352, 52.3037872, -64.8460388, 59.4299088

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -25.7230053, 91.4297180, -22.2335720, 79.2846985, -105.0077057, 113.6632843
1: -15.8495493, 55.2890701, -13.8561430, 46.4380569, -62.2876053, 69.1452026
2: -8.6752014, 51.3622665, -7.6999674, 42.2512589, -50.9264526, 59.0622215
3: -12.3136873, 75.5141220, -10.8138218, 64.4220734, -76.7357559, 86.3279343
4: -16.5161533, 62.5906906, -14.6088352, 52.3037872, -68.8199387, 77.1995239

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -28.7402172, 101.8141479, -118.9864655, 88.8690414
1: -10.3972998, 34.8535805, -17.8190517, 61.8742256, -72.2714996, 52.6726303
2: -5.7887311, 31.5996094, -9.7923040, 57.4433746, -63.2321053, 41.3918991
3: -8.2676754, 48.3058929, -13.7387075, 84.5946655, -92.8623428, 62.0446014
4: -10.9508162, 39.1615753, -18.6144142, 70.0630035, -81.0138168, 57.7759895

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -19.3333740, 68.2948456, -28.5969753, 101.2977753, -120.6311493, 96.8918228
1: -11.9587393, 40.0924377, -17.7317047, 61.6179695, -73.5766983, 57.8241425
2: -6.6578412, 36.4354782, -9.7460089, 57.2178841, -63.8757248, 46.1814880
3: -9.4542150, 55.6175156, -13.6763706, 84.2331619, -93.6873627, 69.2938690
4: -12.6182709, 45.1233826, -18.5252819, 69.7721786, -82.3904495, 63.6486664

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -41.4471855, 153.0721283, -170.2444458, 101.5759964
1: -10.3972998, 34.8535805, -26.8247814, 89.2847366, -99.6820297, 61.6783600
2: -5.7887311, 31.5996094, -14.6021109, 81.6006775, -87.3894119, 46.2017174
3: -8.2676754, 48.3058929, -20.3060989, 123.8886642, -132.1563263, 68.6119919
4: -10.9508162, 39.1615753, -27.8461246, 100.9379883, -111.8888016, 67.0076981

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8885061, upper bound: 147.8727621
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9036995, upper bound: 147.9025467
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -19.3333740, 68.2948456, -40.7301178, 150.6075439, -169.9409180, 109.0249634
1: -11.9587393, 40.0924377, -26.3701305, 87.6974716, -99.6562119, 66.4625702
2: -6.6578412, 36.4354782, -14.3497448, 80.1287766, -86.7866211, 50.7852211
3: -9.4542150, 55.6175156, -19.9721146, 121.7275925, -131.1818085, 75.5896301
4: -12.6182709, 45.1233826, -27.3573246, 99.1469650, -111.7652283, 72.4807053

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8939400, upper bound: 147.8930500
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9034304, upper bound: 147.9024002
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -25.5205307, 90.3732300, -28.2632713, 100.0189743, -125.5395050, 118.6364975
1: -15.6282148, 54.7869644, -17.4716244, 60.8550873, -76.4832916, 72.2585907
2: -8.5134039, 51.1229553, -9.5869884, 56.5679207, -65.0813217, 60.7099457
3: -12.0381298, 74.7684860, -13.4547253, 83.1405716, -95.1786957, 88.2232132
4: -16.0789509, 62.1488037, -18.1971512, 68.9150848, -84.9940338, 80.3459473

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8509008
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8758662
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -28.2850590, 100.0496979, -29.1594334, 103.3192902, -131.6043396, 129.2091370
1: -17.4821606, 60.8535385, -18.0970116, 62.8190041, -80.3011627, 78.9505463
2: -9.5946980, 56.5478783, -9.9481611, 58.3167686, -67.9114609, 66.4960327
3: -13.4687023, 83.1488724, -13.9435673, 85.8972778, -99.3659821, 97.0924377
4: -18.2281151, 68.9123154, -18.9120045, 71.1313782, -89.3594742, 87.8243179

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8675183, upper bound: 147.8523902
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8675183, upper bound: 147.9008247
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -25.5205307, 90.3732300, -43.6764946, 162.2193298, -187.7398224, 134.0497284
1: -15.6282148, 54.7869644, -28.3008976, 93.9612732, -109.5894852, 83.0878601
2: -8.5134039, 51.1229553, -15.3738346, 85.8437195, -94.3571091, 66.4967880
3: -12.0381298, 74.7684860, -21.3804073, 130.5597687, -142.5979004, 96.1488953
4: -16.0789509, 62.1488037, -29.2512608, 106.2766418, -122.3555908, 91.4000397

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8509008
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8764277
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -28.2850590, 100.0496979, -45.1296425, 167.5707703, -195.8558350, 145.1793365
1: -17.4821606, 60.8535385, -29.2989826, 97.1900024, -114.6721649, 90.1525116
2: -9.5946980, 56.5478783, -15.9448833, 88.7482910, -98.3429871, 72.4927597
3: -13.4687023, 83.1488724, -22.1569614, 135.0545654, -148.5232544, 105.3058319
4: -18.2281151, 68.9123154, -30.3614540, 109.9190445, -128.1471252, 99.2737732

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8525133, upper bound: 147.8509299
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8525134, upper bound: 147.9023109
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -37.2249680, 137.0619812, -20.5580711, 72.8395386, -110.0644989, 157.6200562
1: -23.6081371, 79.2028275, -12.6048326, 42.4445572, -66.0526962, 91.8076477
2: -12.8477907, 72.7835312, -7.0036321, 38.5909653, -51.4387550, 79.7871628
3: -18.0774994, 109.5535431, -9.8690500, 58.7866478, -76.8641129, 119.4225922
4: -24.2366123, 89.8513870, -13.2769909, 47.7300758, -71.9666672, 103.1283798

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018653
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018653
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -37.6381721, 138.6512604, -20.5580711, 72.8395386, -110.4777069, 159.2093353
1: -24.0115528, 80.4229202, -12.6048326, 42.4445572, -66.4561081, 93.0277328
2: -13.1183281, 73.8105316, -7.0036321, 38.5909653, -51.7092896, 80.8141556
3: -18.4860172, 111.3000565, -9.8690500, 58.7866478, -77.2726288, 121.1690979
4: -24.7948151, 91.2762527, -13.2769909, 47.7300758, -72.5248718, 104.5532455

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018714
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018714
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -37.2249680, 137.0619812, -22.4526672, 80.2389984, -117.4639587, 159.5146484
1: -23.6081371, 79.2028275, -14.0240250, 46.9927139, -70.6008530, 93.2268448
2: -12.8477907, 72.7835312, -7.7921991, 42.7557449, -55.6035347, 80.5757217
3: -18.0774994, 109.5535431, -10.9185658, 65.2038727, -83.2813568, 120.4720917
4: -24.2366123, 89.8513870, -14.7857008, 52.9388504, -77.1754456, 104.6370850

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018068
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018249, upper bound: 147.9018068
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -37.6381721, 138.6512604, -22.4526672, 80.2389984, -117.8771667, 161.1039124
1: -24.0115528, 80.4229202, -14.0240250, 46.9927139, -71.0042648, 94.4469299
2: -13.1183281, 73.8105316, -7.7921991, 42.7557449, -55.8740692, 81.6027145
3: -18.4860172, 111.3000565, -10.9185658, 65.2038727, -83.6898727, 122.2185974
4: -24.7948151, 91.2762527, -14.7857008, 52.9388504, -77.7336502, 106.0619507

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018824
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9018249, upper bound: 147.9018824
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -49.7711372, 183.6781158, -19.9656219, 70.6679306, -120.4390717, 203.5822296
1: -32.3139000, 106.9501190, -12.2162371, 41.1474991, -73.4613953, 119.1663589
2: -17.6600838, 97.5390244, -6.7891793, 37.3934860, -55.0535698, 104.3214569
3: -24.4670029, 148.8292694, -9.5881958, 57.0037689, -81.4707718, 158.4174652
4: -33.5628624, 120.9773407, -12.8647232, 46.2621765, -79.8250198, 133.8420715

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8937515, upper bound: 147.8951272
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8937515, upper bound: 147.8979023
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -46.5662003, 173.3998108, -20.5580711, 72.8395386, -119.4057312, 193.9578857
1: -30.2708492, 100.2623215, -12.6048326, 42.4445572, -72.7154083, 112.8671417
2: -16.4620190, 91.5210800, -7.0036321, 38.5909653, -55.0529785, 98.5247040
3: -22.8761139, 139.4121094, -9.8690500, 58.7866478, -81.6627426, 149.2811584
4: -31.3180485, 113.4150085, -13.2769909, 47.7300758, -79.0481262, 126.6920013

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9035496
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9036074
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -48.8722305, 180.5112762, -21.6069965, 77.0948105, -125.9670334, 202.1182709
1: -31.7322063, 104.9827423, -13.4670362, 45.1653442, -76.8975525, 118.4497757
2: -17.3392315, 95.7287216, -7.4825358, 41.0797005, -58.4189262, 103.2112503
3: -24.0404701, 146.1153564, -10.5108910, 62.6711311, -86.7115860, 156.6262512
4: -32.9471855, 118.7545853, -14.1882782, 50.8659554, -83.8131409, 132.9428558

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8930500, upper bound: 147.8939400
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8930500, upper bound: 147.8980048
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.7032547, 170.3613129, -22.4526672, 80.2389984, -125.9422531, 192.8139801
1: -29.7180710, 98.3908768, -14.0240250, 46.9927139, -76.7107849, 112.4148941
2: -16.1558971, 89.8006821, -7.7921991, 42.7557449, -58.9116440, 97.5928726
3: -22.4663754, 136.8283844, -10.9185658, 65.2038727, -87.6702423, 147.7469482
4: -30.7305450, 111.2951050, -14.7857008, 52.9388504, -83.6693954, 126.0808029

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9034304
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9035858
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -37.1427765, 136.7313538, -28.7402172, 101.8141479, -138.9569244, 165.4715729
1: -23.5530453, 79.0273056, -17.8190517, 61.8742256, -85.4272537, 96.8463593
2: -12.8184738, 72.6245575, -9.7923040, 57.4433746, -70.2618484, 82.4168472
3: -18.0365276, 109.3061371, -13.7387075, 84.5946655, -102.6311951, 123.0448456
4: -24.1819324, 89.6509018, -18.6144142, 70.0630035, -94.2449341, 108.2653046

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020047, upper bound: 147.8999914
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019887, upper bound: 147.8999914
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -37.6381721, 138.6512604, -28.5969753, 101.2977753, -138.9359436, 167.2482147
1: -24.0115528, 80.4229202, -17.7317047, 61.6179695, -85.6295166, 98.1546249
2: -13.1183281, 73.8105316, -9.7460089, 57.2178841, -70.3362122, 83.5565262
3: -18.4860172, 111.3000565, -13.6763706, 84.2331619, -102.7191772, 124.9764023
4: -24.7948151, 91.2762527, -18.5252819, 69.7721786, -94.5669861, 109.8015366

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020047, upper bound: 147.9001723
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020047, upper bound: 147.9001723
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -37.2249680, 137.0619812, -46.8990211, 174.6907501, -211.6085510, 183.7298431
1: -23.6081371, 79.2028275, -30.4879189, 100.9997101, -124.5303116, 109.6907425
2: -12.8477907, 72.7835312, -16.5829163, 92.1960678, -105.0438614, 89.3664474
3: -18.0774994, 109.5535431, -23.0385094, 140.4252625, -158.5027618, 132.5920563
4: -24.2366123, 89.8513870, -31.5534439, 114.2476959, -138.4843140, 121.4048080

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9009292
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9009292
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -37.6381721, 138.6512604, -46.0344353, 171.6449738, -209.0564423, 184.6262207
1: -24.0115528, 80.4229202, -29.9337845, 99.1241913, -123.1131516, 110.3566971
2: -13.1183281, 73.8105316, -16.2762184, 90.4710541, -103.5893860, 90.0867386
3: -18.4860172, 111.3000565, -22.6276932, 137.8368073, -156.3227997, 133.9277191
4: -24.7948151, 91.2762527, -30.9657612, 112.1220551, -136.9168701, 122.2420044

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9011808
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9011945
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -40.2125587, 149.5623322, -26.7845135, 94.4791794, -134.6917419, 176.3468323
1: -26.2476063, 86.4777603, -16.5002308, 57.3844757, -83.6320648, 102.9779587
2: -14.2429361, 78.8518066, -9.0626335, 53.3078918, -67.5508270, 87.9144440
3: -19.8223019, 120.4490433, -12.7394228, 78.4367599, -98.2590485, 133.1762848
4: -27.1005497, 97.8213043, -17.1984730, 64.9927979, -92.0933456, 115.0197601

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8850278, upper bound: 147.8869558
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8850278, upper bound: 147.8954353
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -43.5002861, 161.4406738, -28.9358044, 102.4427032, -145.9429932, 190.3764801
1: -28.2204723, 93.5673523, -17.9471588, 62.3171005, -90.5375748, 111.5144958
2: -15.3433619, 85.4534302, -9.8654280, 57.8613968, -73.2047577, 95.3188553
3: -21.3137112, 130.0486145, -13.8301306, 85.2005081, -106.5142059, 143.8787384
4: -29.1996746, 105.8372116, -18.7492008, 70.5678024, -99.7674713, 124.5864029

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023928, upper bound: 147.9021056
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9023357, upper bound: 147.9014256
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -42.4732742, 158.5088654, -45.2918358, 168.6215057, -210.8424072, 203.4699097
1: -27.7652149, 91.3422623, -29.4423351, 97.3276215, -124.9197769, 120.4983521
2: -15.0638876, 83.2565231, -16.0030022, 88.7875900, -103.8514786, 99.2016449
3: -20.9500999, 127.2925262, -22.2401676, 135.4240723, -156.3550873, 149.3029633
4: -28.6417046, 103.3438644, -30.4316387, 110.0879059, -138.7295837, 133.7754669

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8708840, upper bound: 147.8723167
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8840521, upper bound: 147.8943138
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.8709221, 170.8487549, -46.8713531, 174.5390930, -220.1546021, 217.4870911
1: -29.8116722, 98.7092209, -30.4697857, 100.9477310, -130.6244049, 129.0157318
2: -16.2031250, 90.1152191, -16.5739288, 92.1538849, -108.3570023, 106.6891479
3: -22.4958019, 137.2716980, -23.0207596, 140.3503723, -162.8461609, 160.2743835
4: -30.8160553, 111.6739960, -31.5356617, 114.1921310, -145.0081635, 143.2096558

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8867383, upper bound: 147.8878168
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8962783, upper bound: 147.9051910
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.66 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020029
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020029
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020350
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020355
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019852
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019887
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9019973
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9003966, upper bound: 147.9020047
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020029
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020029
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020350
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020355
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9019926
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8999914, upper bound: 147.9020140
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9019973, upper bound: 147.9005476
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8885061, upper bound: 147.8727621
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9036995, upper bound: 147.9025467
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8939400, upper bound: 147.8930500
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9034304, upper bound: 147.9024002
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8509008
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8758662
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8675183, upper bound: 147.8523902
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8675183, upper bound: 147.9008247
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8509008
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8509008, upper bound: 147.8764277
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8525133, upper bound: 147.8509299
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8525134, upper bound: 147.9023109
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018653
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018653
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018714
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018714
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018068
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018249, upper bound: 147.9018068
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018248, upper bound: 147.9018824
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9018249, upper bound: 147.9018824
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8937515, upper bound: 147.8951272
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8937515, upper bound: 147.8979023
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9035496
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9036074
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8930500, upper bound: 147.8939400
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8930500, upper bound: 147.8980048
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9034304
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023181, upper bound: 147.9035858
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9020047, upper bound: 147.8999914
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9019887, upper bound: 147.8999914
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9020047, upper bound: 147.9001723
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9020047, upper bound: 147.9001723
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9009292
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9009292
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9011808
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023468, upper bound: 147.9011945
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8850278, upper bound: 147.8869558
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8850278, upper bound: 147.8954353
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023928, upper bound: 147.9021056
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.9023357, upper bound: 147.9014256
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8708840, upper bound: 147.8723167
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8840521, upper bound: 147.8943138
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8867383, upper bound: 147.8878168
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -147.8962783, upper bound: 147.9051910

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -17.1723251, 60.1288261, -77.3011398, 77.3011398
1: -10.3972998, 34.8535805, -10.3972998, 34.8535805, -45.2508774, 45.2508774
2: -5.7887311, 31.5996094, -5.7887311, 31.5996094, -37.3883400, 37.3883400
3: -8.2676754, 48.3058929, -8.2676754, 48.3058929, -56.5735664, 56.5735664
4: -10.9508162, 39.1615753, -10.9508162, 39.1615753, -50.1123886, 50.1123886

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -24.2217636, 85.6705246, -102.8428421, 84.3505859
1: -10.3972998, 34.8535805, -14.8323402, 51.1777039, -61.5750046, 49.6859131
2: -5.7887311, 31.5996094, -8.1742229, 47.2538109, -53.0425415, 39.7738304
3: -8.2676754, 48.3058929, -11.4780111, 70.1124115, -78.3800888, 59.7839050
4: -10.9508162, 39.1615753, -15.5008430, 57.7147942, -68.6656113, 54.6624184

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -26.5009613, 93.8624191, -17.1723251, 60.1288261, -86.6297760, 111.0347290
1: -16.3315239, 56.8662643, -10.3972998, 34.8535805, -51.1851006, 67.2635651
2: -8.9596434, 52.8075600, -5.7887311, 31.5996094, -40.5592461, 58.5962906
3: -12.6496582, 77.6712112, -8.2676754, 48.3058929, -60.9555511, 85.9388885
4: -17.0362015, 64.3830643, -10.9508162, 39.1615753, -56.1977730, 75.3338776

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -26.5009613, 93.8624191, -24.2217636, 85.6705246, -112.1714783, 118.0841827
1: -16.3315239, 56.8662643, -14.8323402, 51.1777039, -67.5092316, 71.6986084
2: -8.9596434, 52.8075600, -8.1742229, 47.2538109, -56.2134514, 60.9817810
3: -12.6496582, 77.6712112, -11.4780111, 70.1124115, -82.7620697, 89.1492157
4: -17.0362015, 64.3830643, -15.5008430, 57.7147942, -74.7509918, 79.8839035

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -19.3003674, 68.1737137, -85.3460159, 79.4291916
1: -10.3972998, 34.8535805, -11.9356270, 40.0169449, -50.4142418, 46.7892075
2: -5.7887311, 31.5996094, -6.6434102, 36.3698540, -42.1585846, 38.2430191
3: -8.2676754, 48.3058929, -9.4367895, 55.5120010, -63.7796745, 57.7426796
4: -10.9508162, 39.1615753, -12.5893555, 45.0393105, -55.9901237, 51.7509308

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -25.4648876, 90.3836746, -107.5559998, 85.5937042
1: -10.3972998, 34.8535805, -15.8796959, 54.3403320, -64.7376251, 50.7332687
2: -5.7887311, 31.5996094, -8.7731647, 50.0930061, -55.8817329, 40.3727646
3: -8.2676754, 48.3058929, -12.3943253, 74.6636658, -82.9313431, 60.7002182
4: -10.9508162, 39.1615753, -16.6967964, 61.5094414, -72.4602509, 55.8583717

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -26.4905167, 93.8233032, -19.3003674, 68.1737137, -94.6642303, 113.1236725
1: -16.3241024, 56.8425865, -11.9356270, 40.0169449, -56.3410454, 68.7781982
2: -8.9556246, 52.7859802, -6.6434102, 36.3698540, -45.3254738, 59.4293900
3: -12.6445141, 77.6383209, -9.4367895, 55.5120010, -68.1565170, 87.0750809
4: -17.0282822, 64.3563766, -12.5893555, 45.0393105, -62.0675926, 76.9457321

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -26.5009613, 93.8624191, -25.4648876, 90.3836746, -116.8846359, 119.3272858
1: -16.3315239, 56.8662643, -15.8796959, 54.3403320, -70.6718445, 72.7459564
2: -8.9596434, 52.8075600, -8.7731647, 50.0930061, -59.0526390, 61.5807152
3: -12.6496582, 77.6712112, -12.3943253, 74.6636658, -87.3133240, 90.0655365
4: -17.0362015, 64.3830643, -16.6967964, 61.5094414, -78.5456390, 81.0798492

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -19.2246304, 67.9323654, -17.1723251, 60.1288261, -79.3534546, 85.1046906
1: -11.8859072, 39.8227730, -10.3972998, 34.8535805, -46.7394829, 50.2200699
2: -6.6206121, 36.1757812, -5.7887311, 31.5996094, -38.2202187, 41.9645081
3: -9.4038820, 55.2570419, -8.2676754, 48.3058929, -57.7097740, 63.5247154
4: -12.5422478, 44.8210831, -10.9508162, 39.1615753, -51.7038155, 55.7718964

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -19.2246304, 67.9323654, -24.2217636, 85.6705246, -104.8951569, 92.1541290
1: -11.8859072, 39.8227730, -14.8323402, 51.1777039, -63.0636063, 54.6551094
2: -6.6206121, 36.1757812, -8.1742229, 47.2538109, -53.8744240, 44.3499947
3: -9.4038820, 55.2570419, -11.4780111, 70.1124115, -79.5162964, 66.7350464
4: -12.5422478, 44.8210831, -15.5008430, 57.7147942, -70.2570419, 60.3219261

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.7230053, 91.4297180, -17.1723251, 60.1288261, -85.8518295, 108.6020355
1: -15.8495493, 55.2890701, -10.3972998, 34.8535805, -50.7031288, 65.6863556
2: -8.6752014, 51.3622665, -5.7887311, 31.5996094, -40.2748070, 57.1509972
3: -12.3136873, 75.5141220, -8.2676754, 48.3058929, -60.6195793, 83.7817917
4: -16.5161533, 62.5906906, -10.9508162, 39.1615753, -55.6777229, 73.5415039

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.7230053, 91.4297180, -24.2217636, 85.6705246, -111.3935318, 115.6514816
1: -15.8495493, 55.2890701, -14.8323402, 51.1777039, -67.0272522, 70.1214066
2: -8.6752014, 51.3622665, -8.1742229, 47.2538109, -55.9290123, 59.5364876
3: -12.3136873, 75.5141220, -11.4780111, 70.1124115, -82.4261017, 86.9921265
4: -16.5161533, 62.5906906, -15.5008430, 57.7147942, -74.2309494, 78.0915222

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -19.2246304, 67.9323654, -19.3333740, 68.2948456, -87.5194778, 87.2657394
1: -11.8859072, 39.8227730, -11.9587393, 40.0924377, -51.9783401, 51.7815132
2: -6.6206121, 36.1757812, -6.6578412, 36.4354782, -43.0560913, 42.8336105
3: -9.4038820, 55.2570419, -9.4542150, 55.6175156, -65.0213928, 64.7112503
4: -12.5422478, 44.8210831, -12.6182709, 45.1233826, -57.6656265, 57.4393539

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -19.2246304, 67.9323654, -25.4648876, 90.3836746, -109.6083069, 93.3972549
1: -11.8859072, 39.8227730, -15.8796959, 54.3403320, -66.2262344, 55.7024651
2: -6.6206121, 36.1757812, -8.7731647, 50.0930061, -56.7136116, 44.9489326
3: -9.4038820, 55.2570419, -12.3943253, 74.6636658, -84.0675507, 67.6513672
4: -12.5422478, 44.8210831, -16.6967964, 61.5094414, -74.0516815, 61.5178795

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -25.7230053, 91.4297180, -19.3333740, 68.2948456, -94.0178528, 110.7630920
1: -15.8495493, 55.2890701, -11.9587393, 40.0924377, -55.9419861, 67.2477951
2: -8.6752014, 51.3622665, -6.6578412, 36.4354782, -45.1106758, 58.0200996
3: -12.3136873, 75.5141220, -9.4542150, 55.6175156, -67.9312057, 84.9683151
4: -16.5161533, 62.5906906, -12.6182709, 45.1233826, -61.6395302, 75.2089462

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -25.7230053, 91.4297180, -25.4648876, 90.3836746, -116.1066818, 116.8945923
1: -15.8495493, 55.2890701, -15.8796959, 54.3403320, -70.1898804, 71.1687469
2: -8.6752014, 51.3622665, -8.7731647, 50.0930061, -58.7681999, 60.1354179
3: -12.3136873, 75.5141220, -12.3943253, 74.6636658, -86.9773560, 87.9084473
4: -16.5161533, 62.5906906, -16.6967964, 61.5094414, -78.0255814, 79.2874680

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -26.5142479, 93.9043808, -111.0766983, 86.6430588
1: -10.3972998, 34.8535805, -16.3395844, 56.8917351, -67.2890320, 51.1931648
2: -5.7887311, 31.5996094, -8.9644785, 52.8301125, -58.6188431, 40.5640869
3: -8.2676754, 48.3058929, -12.6572733, 77.7075195, -85.9751968, 60.9631653
4: -10.9508162, 39.1615753, -17.0450287, 64.4122543, -75.3630676, 56.2066040

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8768014, upper bound: 147.8918812
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020091, upper bound: 147.9002781
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -25.6856670, 91.2902451, -108.4625626, 85.8144836
1: -10.3972998, 34.8535805, -15.8240805, 55.2060509, -65.6033478, 50.6776619
2: -5.7887311, 31.5996094, -8.6600266, 51.2886238, -57.0773544, 40.2596359
3: -8.2676754, 48.3058929, -12.2942753, 75.3984680, -83.6661453, 60.6001663
4: -10.9508162, 39.1615753, -16.4859619, 62.4977455, -73.4485626, 55.6475372

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8768014, upper bound: 147.8918812
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9020091, upper bound: 147.9002781
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19.3333740, 68.2948456, -26.5034904, 93.8643188, -113.1976929, 94.7983398
1: -11.9587393, 40.0924377, -16.3319759, 56.8674889, -68.8262100, 56.4244156
2: -6.6578412, 36.4354782, -8.9603529, 52.8080215, -59.4658623, 45.3958282
3: -9.4542150, 55.6175156, -12.6519566, 77.6738205, -87.1280136, 68.2694550
4: -12.6182709, 45.1233826, -17.0369129, 64.3849182, -77.0031891, 62.1602898

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8908742, upper bound: 147.8901117
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019552, upper bound: 147.9004913
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19.3333740, 68.2948456, -25.7230053, 91.4297180, -110.7630920, 94.0178528
1: -11.9587393, 40.0924377, -15.8495493, 55.2890701, -67.2477951, 55.9419861
2: -6.6578412, 36.4354782, -8.6752014, 51.3622665, -58.0200996, 45.1106758
3: -9.4542150, 55.6175156, -12.3136873, 75.5141220, -84.9683151, 67.9311981
4: -12.6182709, 45.1233826, -16.5161533, 62.5906906, -75.2089539, 61.6395302

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8908742, upper bound: 147.8931446
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9019552, upper bound: 147.9004913
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16.6382065, 58.2863960, -37.3454666, 137.7809906, -154.4191895, 95.6318665
1: -10.0165071, 33.6592903, -23.8390903, 80.0760117, -90.0925217, 57.4983788
2: -5.5608001, 30.5466557, -12.8201160, 73.5761108, -79.1369019, 43.3667679
3: -7.9714966, 46.6258163, -17.9067211, 110.9526749, -118.9241562, 64.5325165
4: -10.5012188, 37.8159828, -24.2921658, 90.6909332, -101.1921387, 62.1081467

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8615237, upper bound: 147.8571542
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8865226, upper bound: 147.8698956
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17.1723251, 60.1288261, -40.5518799, 149.6822662, -166.8545837, 100.6807098
1: -10.3972998, 34.8535805, -26.2260685, 87.2837524, -97.6810455, 61.0796509
2: -5.7887311, 31.5996094, -14.2689352, 79.7680740, -85.5568085, 45.8685455
3: -8.2676754, 48.3058929, -19.8514557, 121.1186523, -129.3863220, 68.1573486
4: -10.9508162, 39.1615753, -27.2127171, 98.6653137, -109.6161270, 66.3742905

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8825296, upper bound: 147.8933592
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9036224, upper bound: 147.9025090
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -18.6142769, 65.6161346, -42.9899750, 157.6126556, -176.2269287, 108.6061096
1: -11.4891014, 38.5254974, -27.7982445, 92.3204498, -103.8095551, 66.3237381
2: -6.3959918, 34.9832878, -15.1982079, 84.2466431, -90.6426239, 50.1814957
3: -9.1104746, 53.4597778, -21.0948811, 128.2545776, -137.3650360, 74.5546570
4: -12.1144428, 43.3414230, -28.9327927, 104.3407440, -116.4551849, 72.2742004

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -19.3333740, 68.2948456, -40.1945457, 148.5102997, -167.8436737, 108.4893875
1: -11.9587393, 40.0924377, -26.0161037, 86.5202332, -98.4789734, 66.1085434
2: -6.6578412, 36.4354782, -14.1545258, 79.0574799, -85.7153244, 50.5900040
3: -9.4542150, 55.6175156, -19.7073650, 120.0943680, -129.5485687, 75.3248749
4: -12.6182709, 45.1233826, -26.9831676, 97.8160706, -110.4343338, 72.1065369

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -25.5205307, 90.3732300, -25.5205307, 90.3732300, -115.8937607, 115.8937607
1: -15.6282148, 54.7869644, -15.6282148, 54.7869644, -70.4151764, 70.4151764
2: -8.5134039, 51.1229553, -8.5134039, 51.1229553, -59.6363602, 59.6363602
3: -12.0381298, 74.7684860, -12.0381298, 74.7684860, -86.8066177, 86.8066101
4: -16.0789509, 62.1488037, -16.0789509, 62.1488037, -78.2277527, 78.2277527

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8391902, upper bound: 147.8352629
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8339750, upper bound: 147.8339523
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -25.5205307, 90.3732300, -28.2850590, 100.0496979, -125.5702286, 118.6582870
1: -15.6282148, 54.7869644, -17.4821606, 60.8535385, -76.4817352, 72.2691269
2: -8.5134039, 51.1229553, -9.5946980, 56.5478783, -65.0612564, 60.7176514
3: -12.0381298, 74.7684860, -13.4687023, 83.1488724, -95.1869965, 88.2371826
4: -16.0789509, 62.1488037, -18.2281151, 68.9123154, -84.9912643, 80.3768997

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8391902, upper bound: 147.8734271
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8339750, upper bound: 147.8624327
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -28.2850590, 100.0496979, -25.5205307, 90.3732300, -118.6582870, 125.5702286
1: -17.4821606, 60.8535385, -15.6282148, 54.7869644, -72.2691269, 76.4817429
2: -9.5946980, 56.5478783, -8.5134039, 51.1229553, -60.7176514, 65.0612564
3: -13.4687023, 83.1488724, -12.0381298, 74.7684860, -88.2371826, 95.1870041
4: -18.2281151, 68.9123154, -16.0789509, 62.1488037, -80.3768997, 84.9912643

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8414353, upper bound: 147.8344172
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8435019, upper bound: 147.8339524
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -28.2850590, 100.0496979, -28.2850590, 100.0496979, -128.3347626, 128.3347626
1: -17.4821606, 60.8535385, -17.4821606, 60.8535385, -78.3356934, 78.3356934
2: -9.5946980, 56.5478783, -9.5946980, 56.5478783, -66.1425705, 66.1425705
3: -13.4687023, 83.1488724, -13.4687023, 83.1488724, -96.6175766, 96.6175690
4: -18.2281151, 68.9123154, -18.2281151, 68.9123154, -87.1404190, 87.1404190

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8414353, upper bound: 147.8789475
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8435019, upper bound: 147.9003671
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -25.5205307, 90.3732300, -41.2331085, 153.0636749, -178.5841980, 131.6063385
1: -15.6282148, 54.7869644, -26.4822807, 88.5447235, -104.1729355, 81.2692413
2: -8.5134039, 51.1229553, -14.2594223, 81.2101593, -89.7235565, 65.3823700
3: -12.0381298, 74.7684860, -19.8723679, 122.9109344, -134.9490662, 94.6408539
4: -16.0789509, 62.1488037, -27.0091286, 100.2858505, -116.3647995, 89.1579208

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8359268, upper bound: 147.8355761
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8345459, upper bound: 147.8340167
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -25.5205307, 90.3732300, -44.1364365, 163.8096008, -189.3301392, 134.5096588
1: -15.6282148, 54.7869644, -28.6372948, 94.9668884, -110.5950928, 83.4242554
2: -8.5134039, 51.1229553, -15.5791731, 86.7089691, -95.2223663, 66.7021255
3: -12.0381298, 74.7684860, -21.6554070, 131.9837646, -144.0218964, 96.4238892
4: -16.0789509, 62.1488037, -29.6639023, 107.3990326, -123.4779816, 91.8126984

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8359269, upper bound: 147.8742417
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8345459, upper bound: 147.8632427
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -28.2850590, 100.0496979, -41.5159187, 154.1901398, -182.4752045, 141.5656128
1: -17.4821606, 60.8535385, -26.6727543, 89.1599808, -106.6421432, 87.5262680
2: -9.5946980, 56.5478783, -14.3626165, 81.7671280, -91.3618240, 70.9104919
3: -13.4687023, 83.1488724, -20.0139580, 123.7755585, -137.2442474, 103.1628265
4: -18.2281151, 68.9123154, -27.2032356, 100.9844818, -119.2125854, 96.1155472

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8359013, upper bound: 147.8341673
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8370827, upper bound: 147.8339524
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -28.2850590, 100.0496979, -44.4165573, 164.9255219, -193.2105865, 144.4662476
1: -17.4821606, 60.8535385, -28.8257790, 95.5747910, -113.0569534, 89.6793213
2: -9.5946980, 56.5478783, -15.6811237, 87.2599487, -96.8546448, 72.2289963
3: -13.4687023, 83.1488724, -21.7959576, 132.8388519, -146.3075104, 104.9448318
4: -18.2281151, 68.9123154, -29.8552589, 108.0897293, -126.3178329, 98.7675629

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8359013, upper bound: 147.8789478
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8370827, upper bound: 147.9005256
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -34.1515923, 124.7049561, -17.1723251, 60.1288261, -94.2804184, 141.8772888
1: -21.5428772, 72.6282425, -10.3972998, 34.8535805, -56.3964577, 83.0255356
2: -11.7518492, 66.8233261, -5.7887311, 31.5996094, -43.3514595, 72.6120605
3: -16.5365067, 100.2833786, -8.2676754, 48.3058929, -64.8423843, 108.5510559
4: -22.2231007, 82.3381119, -10.9508162, 39.1615753, -61.3846741, 93.2889252

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -37.2249680, 137.0619812, -25.4252796, 89.8209152, -127.0458755, 162.4026642
1: -23.6081371, 79.2028275, -15.6686201, 53.8649635, -77.4730988, 94.8714371
2: -12.8477907, 72.7835312, -8.6337471, 49.7275772, -62.5753632, 81.4172821
3: -18.0774994, 109.5535431, -12.1087141, 73.9143600, -91.9918594, 121.6622543
4: -24.2366123, 89.8513870, -16.3766346, 60.8971291, -85.1337280, 106.2280045

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -35.0429878, 128.2274475, -17.1723251, 60.1288261, -95.1718140, 145.3997803
1: -22.2699413, 74.8788834, -10.3972998, 34.8535805, -57.1235161, 85.2761841
2: -12.1865950, 68.7809906, -5.7887311, 31.5996094, -43.7862053, 74.5697250
3: -17.1914902, 103.4877319, -8.2676754, 48.3058929, -65.4973755, 111.7554016
4: -23.0798664, 84.9489059, -10.9508162, 39.1615753, -62.2414360, 95.8997192

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -37.6381721, 138.6512604, -25.4252796, 89.8209152, -127.4590836, 164.0765381
1: -24.0115528, 80.4229202, -15.6686201, 53.8649635, -77.8765182, 96.0915222
2: -13.1183281, 73.8105316, -8.6337471, 49.7275772, -62.8458900, 82.4442749
3: -18.4860172, 111.3000565, -12.1087141, 73.9143600, -92.4003754, 123.4087677
4: -24.7948151, 91.2762527, -16.3766346, 60.8971291, -85.6919327, 107.6528778

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -34.1990395, 124.8957214, -19.3333740, 68.2948456, -102.4938736, 144.2290955
1: -21.5749054, 72.7293701, -11.9587393, 40.0924377, -61.6673431, 84.6881104
2: -11.7687683, 66.9149704, -6.6578412, 36.4354782, -48.2042389, 73.5728073
3: -16.5604649, 100.4261780, -9.4542150, 55.6175156, -72.1779785, 109.8803787
4: -22.2540779, 82.4537125, -12.6182709, 45.1233826, -67.3774490, 95.0719757

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -37.2249680, 137.0619812, -25.4648876, 90.3836746, -127.6086426, 162.5268707
1: -23.6081371, 79.2028275, -15.8796959, 54.3403320, -77.9484634, 95.0825195
2: -12.8477907, 72.7835312, -8.7731647, 50.0930061, -62.9407883, 81.5566940
3: -18.0774994, 109.5535431, -12.3943253, 74.6636658, -92.7411652, 121.9478683
4: -24.2366123, 89.8513870, -16.6967964, 61.5094414, -85.7460251, 106.5481644

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.0887985, 128.4113159, -19.3333740, 68.2948456, -103.3836365, 147.7446899
1: -22.3007126, 74.9765320, -11.9587393, 40.0924377, -62.3931427, 86.9352722
2: -12.2030230, 68.8695374, -6.6578412, 36.4354782, -48.6385002, 75.5273743
3: -17.2143345, 103.6254807, -9.4542150, 55.6175156, -72.8318481, 113.0796814
4: -23.1097946, 85.0603638, -12.6182709, 45.1233826, -68.2331772, 97.6786270

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -37.6381721, 138.6512604, -25.4648876, 90.3836746, -128.0218506, 164.1161499
1: -24.0115528, 80.4229202, -15.8796959, 54.3403320, -78.3518829, 96.3026047
2: -13.1183281, 73.8105316, -8.7731647, 50.0930061, -63.2113266, 82.5836945
3: -18.4860172, 111.3000565, -12.3943253, 74.6636658, -93.1496811, 123.6943817
4: -24.7948151, 91.2762527, -16.6967964, 61.5094414, -86.3042297, 107.9730377

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -43.6688766, 159.9518585, -16.6351833, 58.1010780, -101.7699585, 176.5870361
1: -28.2304325, 93.8161774, -10.0598688, 33.6888466, -61.9192810, 103.8760452
2: -15.4357538, 85.6354828, -5.6090136, 30.5057850, -45.9415321, 91.2444763
3: -21.4087944, 130.2965546, -8.0281944, 46.7258339, -68.1346283, 138.3247528
4: -29.3905525, 106.0284119, -10.5998840, 37.8470726, -67.2376175, 116.6282883

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -49.7711372, 183.6781158, -23.0995712, 81.4801941, -131.2513275, 206.6105194
1: -32.3139000, 106.9501190, -14.0779972, 48.3512344, -80.6651306, 121.0145721
2: -17.6600838, 97.5390244, -7.7407179, 44.5251503, -62.1852303, 105.2750778
3: -24.4670029, 148.8292694, -10.9018774, 66.3507385, -90.8177414, 159.7311096
4: -33.5628624, 120.9773407, -14.6768951, 54.4570236, -88.0198746, 135.6542358

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -40.9095459, 150.9592896, -17.1723251, 60.1288261, -101.0383759, 168.1316223
1: -26.4685459, 88.1056595, -10.3972998, 34.8535805, -61.3221283, 98.5029602
2: -14.4055862, 80.5283051, -5.7887311, 31.5996094, -46.0051880, 86.3170395
3: -20.0395470, 122.2502670, -8.2676754, 48.3058929, -68.3454361, 130.5179443
4: -27.4695587, 99.6036758, -10.9508162, 39.1615753, -66.6311340, 110.5544891

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9017568, upper bound: 147.9030713
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9017568, upper bound: 147.9035496
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.5662003, 173.3998108, -25.4252796, 89.8209152, -136.3871155, 198.7383728
1: -30.2708492, 100.2623215, -15.6686201, 53.8649635, -84.1358109, 115.9309311
2: -16.4620190, 91.5210800, -8.6337471, 49.7275772, -66.1895828, 100.1548233
3: -22.8761139, 139.4121094, -12.1087141, 73.9143600, -96.7904739, 151.5207977
4: -31.3180485, 113.4150085, -16.3766346, 60.8971291, -92.2151794, 129.7916260

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9017568, upper bound: 147.9032245
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9017568, upper bound: 147.9036074
time: 0.58 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.87 + 416.69 = 420.56 seconds
