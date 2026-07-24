## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.241187818999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890)
1: (-1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385)
2: (-2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155)
3: (-2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479)
4: (-2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695)
5: (-2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987)
6: (-2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276)
7: (-2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247)
8: (-3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120)
9: (-2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.19 + 3.52 = 5.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -4.2840282, upper bound: 4.2840282

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2828605, upper bound: 4.2830378
time: 1.60 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 1.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.20 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.20
Output dim: 8, lower bound: -4.2828605, upper bound: 4.2830378
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.20
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.2352757, 1.8643738, -2.4096687, 2.0148206, -4.2500963, 4.2740426
1: -1.8017280, 1.7118255, -1.9410727, 1.8341657, -3.6358938, 3.6528978
2: -2.2059555, 1.7625470, -2.4231946, 1.8821208, -4.0880761, 4.1857414
3: -2.4719992, 1.4925237, -2.6752136, 1.5974345, -4.0694332, 4.1677370
4: -2.6304567, 1.7897929, -2.8320472, 1.9358228, -4.5662794, 4.6218400
5: -2.0854411, 1.9205116, -2.2663326, 2.0560665, -4.1415071, 4.1868443
6: -1.8782568, 2.1282935, -2.0454202, 2.3011079, -4.1793647, 4.1737137
7: -2.1847844, 2.1578999, -2.3598943, 2.3240304, -4.5088148, 4.5177937
8: -3.0803409, 1.5620990, -3.3761089, 1.7121029, -4.7924423, 4.9382076
9: -1.9560425, 2.0874662, -2.1206036, 2.2545609, -4.2106037, 4.2080698

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 2.24 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 1.46 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.6859634, 2.1864152, -2.3557107, 1.9695348, -4.6554980, 4.5421257
1: -2.1624675, 2.0292344, -1.8983923, 1.7969595, -3.9594269, 3.9276266
2: -2.7670116, 2.0491982, -2.3576131, 1.8452263, -4.6122379, 4.4068112
3: -3.0094893, 1.7467911, -2.6137304, 1.5651306, -4.5746198, 4.3605213
4: -3.1551137, 2.1608493, -2.7705665, 1.8908819, -5.0459957, 4.9314156
5: -2.5614202, 2.2250535, -2.2111328, 2.0145185, -4.5759387, 4.4361863
6: -2.2950575, 2.5095546, -1.9947557, 2.2498562, -4.5449138, 4.5043106
7: -2.6537476, 2.6119380, -2.3064022, 2.2737103, -4.9274578, 4.9183402
8: -3.7780075, 1.7814345, -3.2866185, 1.6665821, -5.4445896, 5.0680532
9: -2.3763266, 2.4497280, -2.0706875, 2.2036154, -4.5799417, 4.5204153

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 1.61 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
time: 1.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.62 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 8, lower bound: -4.2827872, upper bound: 4.2827872

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2.2352757, 1.8643738, -2.2352757, 1.8643738, -4.0996494, 4.0996494
1: -1.8017280, 1.7118255, -1.8017280, 1.7118255, -3.5135536, 3.5135536
2: -2.2059555, 1.7625470, -2.2059555, 1.7625470, -3.9685025, 3.9685025
3: -2.4719992, 1.4925237, -2.4719992, 1.4925237, -3.9645228, 3.9645228
4: -2.6304567, 1.7897929, -2.6304567, 1.7897929, -4.4202495, 4.4202495
5: -2.0854411, 1.9205116, -2.0854411, 1.9205116, -4.0059528, 4.0059528
6: -1.8782568, 2.1282935, -1.8782568, 2.1282935, -4.0065503, 4.0065503
7: -2.1847844, 2.1578999, -2.1847844, 2.1578999, -4.3426843, 4.3426843
8: -3.0803409, 1.5620990, -3.0803409, 1.5620990, -4.6424398, 4.6424398
9: -1.9560425, 2.0874662, -1.9560425, 2.0874662, -4.0435085, 4.0435085

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2637549, upper bound: 4.2689116
time: 8.74 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815273, upper bound: 4.2817153
time: 1.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.2352757, 1.8643738, -2.6859634, 2.1864152, -4.4216909, 4.5503373
1: -1.8017280, 1.7118255, -2.1624675, 2.0292344, -3.8309624, 3.8742929
2: -2.2059555, 1.7625470, -2.7670116, 2.0491982, -4.2551537, 4.5295587
3: -2.4719992, 1.4925237, -3.0094893, 1.7467911, -4.2187901, 4.5020132
4: -2.6304567, 1.7897929, -3.1551137, 2.1608493, -4.7913060, 4.9449067
5: -2.0854411, 1.9205116, -2.5614202, 2.2250535, -4.3104944, 4.4819317
6: -1.8782568, 2.1282935, -2.2950575, 2.5095546, -4.3878117, 4.4233513
7: -2.1847844, 2.1578999, -2.6537476, 2.6119380, -4.7967224, 4.8116474
8: -3.0803409, 1.5620990, -3.7780075, 1.7814345, -4.8617754, 5.3401065
9: -1.9560425, 2.0874662, -2.3763266, 2.4497280, -4.4057703, 4.4637928

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2637549, upper bound: 4.2689116
time: 2.46 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815273, upper bound: 4.2817153
time: 1.33 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.6859634, 2.1864152, -2.2352757, 1.8643738, -4.5503373, 4.4216909
1: -2.1624675, 2.0292344, -1.8017280, 1.7118255, -3.8742929, 3.8309624
2: -2.7670116, 2.0491982, -2.2059555, 1.7625470, -4.5295587, 4.2551537
3: -3.0094893, 1.7467911, -2.4719992, 1.4925237, -4.5020132, 4.2187901
4: -3.1551137, 2.1608493, -2.6304567, 1.7897929, -4.9449067, 4.7913060
5: -2.5614202, 2.2250535, -2.0854411, 1.9205116, -4.4819317, 4.3104944
6: -2.2950575, 2.5095546, -1.8782568, 2.1282935, -4.4233513, 4.3878117
7: -2.6537476, 2.6119380, -2.1847844, 2.1578999, -4.8116474, 4.7967224
8: -3.7780075, 1.7814345, -3.0803409, 1.5620990, -5.3401065, 4.8617754
9: -2.3763266, 2.4497280, -1.9560425, 2.0874662, -4.4637928, 4.4057703

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2672670, upper bound: 4.2623143
time: 2.06 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2814305, upper bound: 4.2814305
time: 1.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.6859634, 2.1864152, -2.6817932, 2.1864152, -4.8723783, 4.8682084
1: -2.1624675, 2.0292344, -2.1612973, 2.0281916, -4.1906590, 4.1905317
2: -2.7670116, 2.0491982, -2.7636824, 2.0486693, -4.8156810, 4.8128805
3: -3.0094893, 1.7467911, -3.0052636, 1.7463536, -4.7558432, 4.7520547
4: -3.1551137, 2.1608493, -3.1551137, 2.1573672, -5.3124809, 5.3159628
5: -2.5614202, 2.2250535, -2.5613899, 2.2213860, -4.7828064, 4.7864437
6: -2.2950575, 2.5095546, -2.2918289, 2.5083256, -4.8033829, 4.8013835
7: -2.6537476, 2.6119380, -2.6512446, 2.6118093, -5.2655568, 5.2631826
8: -3.7780075, 1.7814345, -3.7768519, 1.7814345, -5.5594420, 5.5582867
9: -2.3763266, 2.4497280, -2.3763266, 2.4479184, -4.8242450, 4.8260546

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2623143, upper bound: 4.2672670
time: 2.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2814305, upper bound: 4.2814305
time: 1.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.41 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2637549, upper bound: 4.2689116
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2815273, upper bound: 4.2817153
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2637549, upper bound: 4.2689116
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2815273, upper bound: 4.2817153
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2672670, upper bound: 4.2623143
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2814305, upper bound: 4.2814305
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2623143, upper bound: 4.2672670
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 8, lower bound: -4.2814305, upper bound: 4.2814305

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.6751651, 1.4680865, -2.0726666, 1.7400761, -3.4152412, 3.5407531
1: -1.3563435, 1.3210368, -1.6699942, 1.5965317, -2.9528751, 2.9910312
2: -1.5323389, 1.4056618, -2.0104589, 1.6563345, -3.1886735, 3.4161208
3: -1.8117030, 1.1672130, -2.2809269, 1.3960905, -3.2077935, 3.4481399
4: -1.9690716, 1.3418083, -2.4337058, 1.6565086, -3.6255803, 3.7755141
5: -1.4871384, 1.5101548, -1.9131263, 1.7991836, -3.2863221, 3.4232812
6: -1.3727816, 1.6423122, -1.7261449, 1.9831450, -3.3559265, 3.3684571
7: -1.6208106, 1.6182514, -2.0216887, 2.0003862, -3.6211967, 3.6399403
8: -2.1370544, 1.2230477, -2.8135865, 1.4551802, -3.5922346, 4.0366344
9: -1.4300731, 1.5040156, -1.8021669, 1.9206136, -3.3506868, 3.3061824

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2620317, upper bound: 4.2669284
time: 2.50 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668866
time: 2.11 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.1039996, 1.7637614, -2.2352757, 1.8643738, -3.9683733, 3.9990373
1: -1.6962105, 1.6183649, -1.8017280, 1.7118255, -3.4080360, 3.4200931
2: -2.0477369, 1.6768968, -2.2059555, 1.7625470, -3.8102839, 3.8828523
3: -2.3186967, 1.4140369, -2.4719992, 1.4925237, -3.8112204, 3.8860359
4: -2.4741194, 1.6818365, -2.6304567, 1.7897929, -4.2639122, 4.3122931
5: -1.9457788, 1.8224949, -2.0854411, 1.9205116, -3.8662906, 3.9079361
6: -1.7551769, 2.0107517, -1.8782568, 2.1282935, -3.8834705, 3.8890085
7: -2.0534024, 2.0305119, -2.1847844, 2.1578999, -4.2113023, 4.2152963
8: -2.8671131, 1.4764357, -3.0803409, 1.5620990, -4.4292121, 4.5567765
9: -1.8330430, 1.9621764, -1.9560425, 2.0874662, -3.9205093, 3.9182191

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 20

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2664451, upper bound: 4.2664900
time: 4.85 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2821020, upper bound: 4.2821020
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6751651, 1.4680865, -2.5292394, 2.0649080, -3.7400732, 3.9973259
1: -1.3563435, 1.3210368, -2.0357287, 1.9192511, -3.2755947, 3.3567655
2: -1.5323389, 1.4056618, -2.5779083, 1.9460571, -3.4783959, 3.9835701
3: -1.8117030, 1.1672130, -2.8255737, 1.6551715, -3.4668746, 3.9927866
4: -1.9690716, 1.3418083, -2.9685242, 2.0305922, -3.9996638, 4.3103323
5: -1.4871384, 1.5101548, -2.3983636, 2.1063170, -3.5934553, 3.9085183
6: -1.3727816, 1.6423122, -2.1480584, 2.3674090, -3.7401905, 3.7903705
7: -1.6208106, 1.6182514, -2.4956448, 2.4612384, -4.0820489, 4.1138964
8: -2.1370544, 1.2230477, -3.5283895, 1.6754414, -3.8124957, 4.7514372
9: -1.4300731, 1.5040156, -2.2291350, 2.2898922, -3.7199655, 3.7331505

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2605059, upper bound: 4.2658898
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.1039996, 1.7637614, -2.6859634, 2.1864152, -4.2904148, 4.4497247
1: -1.6962105, 1.6183649, -2.1624675, 2.0292344, -3.7254448, 3.7808323
2: -2.0477369, 1.6768968, -2.7670116, 2.0491982, -4.0969353, 4.4439087
3: -2.3186967, 1.4140369, -3.0094893, 1.7467911, -4.0654879, 4.4235263
4: -2.4741194, 1.6818365, -3.1551137, 2.1608493, -4.6349688, 4.8369503
5: -1.9457788, 1.8224949, -2.5614202, 2.2250535, -4.1708326, 4.3839149
6: -1.7551769, 2.0107517, -2.2950575, 2.5095546, -4.2647314, 4.3058090
7: -2.0534024, 2.0305119, -2.6537476, 2.6119380, -4.6653404, 4.6842594
8: -2.8671131, 1.4764357, -3.7780075, 1.7814345, -4.6485476, 5.2544432
9: -1.8330430, 1.9621764, -2.3763266, 2.4497280, -4.2827711, 4.3385029

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2629085, upper bound: 4.2632023
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2814761, upper bound: 4.2816645
time: 8.20 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2.5292394, 2.0649080, -1.6751651, 1.4680865, -3.9973259, 3.7400732
1: -2.0357287, 1.9192511, -1.3563435, 1.3210368, -3.3567655, 3.2755947
2: -2.5779083, 1.9460571, -1.5323389, 1.4056618, -3.9835701, 3.4783959
3: -2.8255737, 1.6551715, -1.8117030, 1.1672130, -3.9927866, 3.4668746
4: -2.9685242, 2.0305922, -1.9690716, 1.3418083, -4.3103323, 3.9996638
5: -2.3983636, 2.1063170, -1.4871384, 1.5101548, -3.9085183, 3.5934553
6: -2.1480584, 2.3674090, -1.3727816, 1.6423122, -3.7903705, 3.7401905
7: -2.4956448, 2.4612384, -1.6208106, 1.6182514, -4.1138964, 4.0820489
8: -3.5283895, 1.6754414, -2.1370544, 1.2230477, -4.7514372, 3.8124957
9: -2.2291350, 2.2898922, -1.4300731, 1.5040156, -3.7331505, 3.7199655

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2658898, upper bound: 4.2605059
time: 1.82 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
time: 2.72 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.6859634, 2.1864152, -2.1039996, 1.7637614, -4.4497247, 4.2904148
1: -2.1624675, 2.0292344, -1.6962105, 1.6183649, -3.7808323, 3.7254448
2: -2.7670116, 2.0491982, -2.0477369, 1.6768968, -4.4439087, 4.0969353
3: -3.0094893, 1.7467911, -2.3186967, 1.4140369, -4.4235263, 4.0654879
4: -3.1551137, 2.1608493, -2.4741194, 1.6818365, -4.8369503, 4.6349688
5: -2.5614202, 2.2250535, -1.9457788, 1.8224949, -4.3839149, 4.1708326
6: -2.2950575, 2.5095546, -1.7551769, 2.0107517, -4.3058090, 4.2647314
7: -2.6537476, 2.6119380, -2.0534024, 2.0305119, -4.6842594, 4.6653404
8: -3.7780075, 1.7814345, -2.8671131, 1.4764357, -5.2544432, 4.6485476
9: -2.3763266, 2.4497280, -1.8330430, 1.9621764, -4.3385029, 4.2827711

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2632023, upper bound: 4.2629085
time: 6.02 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2816645, upper bound: 4.2814761
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0913451, 1.7484686, -2.5269723, 2.0649080, -4.1562529, 4.2754412
1: -1.6824763, 1.6148361, -2.0350785, 1.9188769, -3.6013532, 3.6499147
2: -2.0444131, 1.6619596, -2.5760598, 1.9458667, -3.9902797, 4.2380195
3: -2.3074937, 1.4022710, -2.8232300, 1.6550149, -3.9625087, 4.2255011
4: -2.4546425, 1.6726630, -2.9685242, 2.0290921, -4.4837346, 4.6411872
5: -1.9334565, 1.7881005, -2.3983524, 2.1042707, -4.0377274, 4.1864529
6: -1.7411426, 1.9812360, -2.1462605, 2.3669646, -4.1081071, 4.1274967
7: -2.0472083, 2.0340009, -2.4942563, 2.4611926, -4.5084009, 4.5282574
8: -2.8049994, 1.3906354, -3.5278225, 1.6754414, -4.4804406, 4.9184580
9: -1.8189635, 1.8345429, -2.2291350, 2.2889993, -4.1079626, 4.0636778

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2558295, upper bound: 4.2628669
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.5530794, 2.0842085, -2.6817932, 2.1864152, -4.7394943, 4.7660017
1: -2.0557983, 1.9356909, -2.1612973, 2.0281916, -4.0839901, 4.0969882
2: -2.6061571, 1.9619758, -2.7636824, 2.0486693, -4.6548262, 4.7256584
3: -2.8541253, 1.6687641, -3.0052636, 1.7463536, -4.6004791, 4.6740274
4: -2.9991403, 2.0503845, -3.1551137, 2.1573672, -5.1565075, 5.2054982
5: -2.4226506, 2.1249256, -2.5613899, 2.2213860, -4.6440363, 4.6863155
6: -2.1705265, 2.3895175, -2.2918289, 2.5083256, -4.6788521, 4.6813464
7: -2.5197325, 2.4836340, -2.6512446, 2.6118093, -5.1315417, 5.1348786
8: -3.5676296, 1.6942568, -3.7768519, 1.7814345, -5.3490639, 5.4711084
9: -2.2525322, 2.3232732, -2.3763266, 2.4479184, -4.7004509, 4.6995997

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799981, upper bound: 4.2797895
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
time: 2.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.04 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2620317, upper bound: 4.2669284
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668866
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2664451, upper bound: 4.2664900
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2821020, upper bound: 4.2821020
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2605059, upper bound: 4.2658898
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2629085, upper bound: 4.2632023
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2814761, upper bound: 4.2816645
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2658898, upper bound: 4.2605059
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2632023, upper bound: 4.2629085
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2816645, upper bound: 4.2814761
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2558295, upper bound: 4.2628669
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2799981, upper bound: 4.2797895
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.04
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.6751651, 1.4680865, -1.9366288, 1.6337672, -3.3089323, 3.4047153
1: -1.3563435, 1.3210368, -1.5673178, 1.5009624, -2.8573060, 2.8883548
2: -1.5323389, 1.4056618, -1.8507612, 1.5691757, -3.1015146, 3.2564230
3: -1.8117030, 1.1672130, -2.1284184, 1.3194178, -3.1311207, 3.2956314
4: -1.9690716, 1.3418083, -2.2774007, 1.5524859, -3.5215576, 3.6192091
5: -1.4871384, 1.5101548, -1.7649215, 1.6867763, -3.1739147, 3.2750764
6: -1.3727816, 1.6423122, -1.5985570, 1.8565005, -3.2292821, 3.2408690
7: -1.6208106, 1.6182514, -1.8922317, 1.8724862, -3.4932969, 3.5104833
8: -2.1370544, 1.2230477, -2.5962846, 1.3617575, -3.4988120, 3.8193324
9: -1.4300731, 1.5040156, -1.6760186, 1.7979292, -3.2280023, 3.1800342

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668861
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668861
time: 2.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.6515318, 1.4511701, -2.1927042, 1.8270174, -3.4785492, 3.6438742
1: -1.3393404, 1.3055245, -1.7667007, 1.6783012, -3.0176415, 3.0722251
2: -1.5071620, 1.3910279, -2.1604946, 1.7222885, -3.2294505, 3.5515225
3: -1.7849113, 1.1539679, -2.4245210, 1.4634503, -3.2483616, 3.5784888
4: -1.9439080, 1.3244337, -2.5693102, 1.7527267, -3.6966348, 3.8937440
5: -1.4614565, 1.4906766, -2.0367942, 1.8529298, -3.3143864, 3.5274708
6: -1.3530947, 1.6205612, -1.8287529, 2.0667644, -3.4198589, 3.4493141
7: -1.5997339, 1.5962027, -2.1492403, 2.1234856, -3.7232194, 3.7454429
8: -2.0984631, 1.2120751, -2.9839160, 1.4802775, -3.5787406, 4.1959910
9: -1.4087726, 1.4862524, -1.9085587, 1.9913862, -3.4001589, 3.3948112

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668866
time: 2.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668866
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.0791783, 1.7451546, -2.2065530, 1.8596144, -3.9387927, 3.9517076
1: -1.6767284, 1.6008706, -1.7857372, 1.6896710, -3.3663993, 3.3866076
2: -2.0177283, 1.6610575, -2.1739576, 1.7406670, -3.7583952, 3.8350151
3: -2.2898307, 1.3998376, -2.4433260, 1.4767292, -3.7665598, 3.8431635
4: -2.4449968, 1.6617584, -2.5955205, 1.7650933, -4.2100902, 4.2572789
5: -1.9193356, 1.8047465, -2.0571856, 1.8935564, -3.8128920, 3.8619323
6: -1.7319556, 1.9891934, -1.8475021, 2.1045749, -3.8365307, 3.8366957
7: -2.0286033, 2.0061741, -2.1588426, 2.1302233, -4.1588268, 4.1650167
8: -2.8269677, 1.4623269, -3.0295806, 1.5362818, -4.3632498, 4.4919076
9: -1.8098855, 1.9406712, -1.9258299, 2.0635457, -3.8734312, 3.8665011

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.1930698, upper bound: 4.2421890
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2649519, upper bound: 4.2646646
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.1039996, 1.7637614, -2.2224200, 1.8544399, -3.9584394, 3.9861813
1: -1.6962105, 1.6183649, -1.7916270, 1.7027761, -3.3989866, 3.4099920
2: -2.0477369, 1.6768968, -2.1904099, 1.7542399, -3.8019767, 3.8673067
3: -2.3186967, 1.4140369, -2.4569645, 1.4852233, -3.8039200, 3.8710012
4: -2.4741194, 1.6818365, -2.6153052, 1.7793736, -4.2534933, 4.2971416
5: -1.9457788, 1.8224949, -2.0717945, 1.9111882, -3.8569670, 3.8942895
6: -1.7551769, 2.0107517, -1.8660779, 2.1168451, -3.8720222, 3.8768296
7: -2.0534024, 2.0305119, -2.1719918, 2.1453404, -4.1987429, 4.2025037
8: -2.8671131, 1.4764357, -3.0597453, 1.5539397, -4.4210529, 4.5361810
9: -1.8330430, 1.9621764, -1.9440522, 2.0761576, -3.9092007, 3.9062285

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808662, upper bound: 4.2807667
time: 3.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6751651, 1.4680865, -2.3968477, 1.9568636, -3.6320286, 3.8649342
1: -1.3563435, 1.3210368, -1.9351065, 1.8265558, -3.1828995, 3.2561433
2: -1.5323389, 1.4056618, -2.4212003, 1.8581847, -3.3905234, 3.8268621
3: -1.8117030, 1.1672130, -2.6757462, 1.5809891, -3.3926921, 3.8429592
4: -1.9690716, 1.3418083, -2.8170159, 1.9246368, -3.8937085, 4.1588240
5: -1.4871384, 1.5101548, -2.2567866, 1.9988856, -3.4860239, 3.7669415
6: -1.3727816, 1.6423122, -2.0223429, 2.2426498, -3.6154313, 3.6646552
7: -1.6208106, 1.6182514, -2.3676341, 2.3367693, -3.9575801, 3.9858856
8: -2.1370544, 1.2230477, -3.3195601, 1.5786661, -3.7157207, 4.5426078
9: -1.4300731, 1.5040156, -2.1067595, 2.1695790, -3.5996523, 3.6107750

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6515318, 1.4511701, -2.6570215, 2.1486967, -3.8002286, 4.1081915
1: -1.3393404, 1.3055245, -2.1382222, 2.0060167, -3.3453572, 3.4437466
2: -1.5071620, 1.3910279, -2.7360160, 2.0174913, -3.5246534, 4.1270437
3: -1.7849113, 1.1539679, -2.9781241, 1.7255940, -3.5105052, 4.1320920
4: -1.9439080, 1.3244337, -3.1112006, 2.1333952, -4.0773029, 4.4356341
5: -1.4614565, 1.4906766, -2.5278871, 2.1722200, -3.6336765, 4.0185637
6: -1.3530947, 1.6205612, -2.2574952, 2.4575932, -3.8106880, 3.8780565
7: -1.5997339, 1.5962027, -2.6308317, 2.5906546, -4.1903887, 4.2270346
8: -2.0984631, 1.2120751, -3.7043788, 1.7051127, -3.8035758, 4.9164538
9: -1.4087726, 1.4862524, -2.3408747, 2.3680873, -3.7768598, 3.8271270

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.0914512, 1.7711089, -2.6594906, 2.1662016, -4.2576528, 4.4305992
1: -1.6925064, 1.6076061, -2.1417589, 2.0106840, -3.7031903, 3.7493649
2: -2.0345435, 1.6656239, -2.7349446, 2.0321817, -4.0667253, 4.4005685
3: -2.3086519, 1.4077268, -2.9786813, 1.7316976, -4.0403495, 4.3864079
4: -2.4584424, 1.6701031, -3.1245277, 2.1392226, -4.5976648, 4.7946310
5: -1.9332569, 1.8058885, -2.5339155, 2.2057736, -4.1390305, 4.3398042
6: -1.7391957, 2.0005102, -2.2703793, 2.4862311, -4.2254267, 4.2708893
7: -2.0426583, 2.0184693, -2.6271253, 2.5862570, -4.6289153, 4.6455946
8: -2.8418233, 1.4601251, -3.7366238, 1.7656337, -4.6074572, 5.1967487
9: -1.8173971, 1.9517206, -2.3519406, 2.4268391, -4.2442360, 4.3036613

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1759292, upper bound: 4.1240043
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2606964, upper bound: 4.2614438
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2.0913379, 1.7541999, -2.6859634, 2.1864152, -4.2777529, 4.4401631
1: -1.6863245, 1.6094422, -2.1624675, 2.0292344, -3.7155590, 3.7719097
2: -2.0324769, 1.6687468, -2.7670116, 2.0491982, -4.0816751, 4.4357586
3: -2.3040586, 1.4067905, -3.0094893, 1.7467911, -4.0508499, 4.4162798
4: -2.4592841, 1.6715741, -3.1551137, 2.1608493, -4.6201334, 4.8266878
5: -1.9322655, 1.8132346, -2.5614202, 2.2250535, -4.1573191, 4.3746548
6: -1.7432476, 1.9995810, -2.2950575, 2.5095546, -4.2528024, 4.2946386
7: -2.0408397, 2.0181684, -2.6537476, 2.6119380, -4.6527777, 4.6719160
8: -2.8465054, 1.4688146, -3.7780075, 1.7814345, -4.6279402, 5.2468224
9: -1.8212051, 1.9510278, -2.3763266, 2.4497280, -4.2709332, 4.3273544

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2800680, upper bound: 4.2802284
time: 2.74 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797491, upper bound: 4.2801302
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2.3968477, 1.9568636, -1.6751651, 1.4680865, -3.8649342, 3.6320286
1: -1.9351065, 1.8265558, -1.3563435, 1.3210368, -3.2561433, 3.1828995
2: -2.4212003, 1.8581847, -1.5323389, 1.4056618, -3.8268621, 3.3905234
3: -2.6757462, 1.5809891, -1.8117030, 1.1672130, -3.8429592, 3.3926921
4: -2.8170159, 1.9246368, -1.9690716, 1.3418083, -4.1588240, 3.8937085
5: -2.2567866, 1.9988856, -1.4871384, 1.5101548, -3.7669415, 3.4860239
6: -2.0223429, 2.2426498, -1.3727816, 1.6423122, -3.6646552, 3.6154313
7: -2.3676341, 2.3367693, -1.6208106, 1.6182514, -3.9858856, 3.9575801
8: -3.3195601, 1.5786661, -2.1370544, 1.2230477, -4.5426078, 3.7157207
9: -2.1067595, 2.1695790, -1.4300731, 1.5040156, -3.6107750, 3.5996523

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
time: 1.68 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2.6570215, 2.1486967, -1.6515318, 1.4511701, -4.1081915, 3.8002286
1: -2.1382222, 2.0060167, -1.3393404, 1.3055245, -3.4437466, 3.3453572
2: -2.7360160, 2.0174913, -1.5071620, 1.3910279, -4.1270437, 3.5246534
3: -2.9781241, 1.7255940, -1.7849113, 1.1539679, -4.1320920, 3.5105052
4: -3.1112006, 2.1333952, -1.9439080, 1.3244337, -4.4356341, 4.0773029
5: -2.5278871, 2.1722200, -1.4614565, 1.4906766, -4.0185637, 3.6336765
6: -2.2574952, 2.4575932, -1.3530947, 1.6205612, -3.8780565, 3.8106880
7: -2.6308317, 2.5906546, -1.5997339, 1.5962027, -4.2270346, 4.1903887
8: -3.7043788, 1.7051127, -2.0984631, 1.2120751, -4.9164538, 3.8035758
9: -2.3408747, 2.3680873, -1.4087726, 1.4862524, -3.8271270, 3.7768598

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
time: 1.69 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -2.6594906, 2.1662016, -2.0914512, 1.7711089, -4.4305992, 4.2576528
1: -2.1417589, 2.0106840, -1.6925064, 1.6076061, -3.7493649, 3.7031903
2: -2.7349446, 2.0321817, -2.0345435, 1.6656239, -4.4005685, 4.0667253
3: -2.9786813, 1.7316976, -2.3086519, 1.4077268, -4.3864079, 4.0403495
4: -3.1245277, 2.1392226, -2.4584424, 1.6701031, -4.7946310, 4.5976648
5: -2.5339155, 2.2057736, -1.9332569, 1.8058885, -4.3398042, 4.1390305
6: -2.2703793, 2.4862311, -1.7391957, 2.0005102, -4.2708893, 4.2254267
7: -2.6271253, 2.5862570, -2.0426583, 2.0184693, -4.6455946, 4.6289153
8: -3.7366238, 1.7656337, -2.8418233, 1.4601251, -5.1967487, 4.6074572
9: -2.3519406, 2.4268391, -1.8173971, 1.9517206, -4.3036613, 4.2442360

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1240043, upper bound: 4.1759292
time: 2.11 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2614438, upper bound: 4.2606964
time: 2.35 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -2.6859634, 2.1864152, -2.0913379, 1.7541999, -4.4401631, 4.2777529
1: -2.1624675, 2.0292344, -1.6863245, 1.6094422, -3.7719097, 3.7155590
2: -2.7670116, 2.0491982, -2.0324769, 1.6687468, -4.4357586, 4.0816751
3: -3.0094893, 1.7467911, -2.3040586, 1.4067905, -4.4162798, 4.0508499
4: -3.1551137, 2.1608493, -2.4592841, 1.6715741, -4.8266878, 4.6201334
5: -2.5614202, 2.2250535, -1.9322655, 1.8132346, -4.3746548, 4.1573191
6: -2.2950575, 2.5095546, -1.7432476, 1.9995810, -4.2946386, 4.2528024
7: -2.6537476, 2.6119380, -2.0408397, 2.0181684, -4.6719160, 4.6527777
8: -3.7780075, 1.7814345, -2.8465054, 1.4688146, -5.2468224, 4.6279402
9: -2.3763266, 2.4497280, -1.8212051, 1.9510278, -4.3273544, 4.2709332

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2802284, upper bound: 4.2800680
time: 1.51 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797491
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.0913451, 1.7484686, -2.3962343, 1.9568636, -4.0482087, 4.1447029
1: -1.6824763, 1.6148361, -1.9349238, 1.8265558, -3.5090322, 3.5497599
2: -2.0444131, 1.6619596, -2.4206822, 1.8581847, -3.9025979, 4.0826416
3: -2.3074937, 1.4022710, -2.6750891, 1.5809891, -3.8884828, 4.0773602
4: -2.4546425, 1.6726630, -2.8170159, 1.9244330, -4.3790755, 4.4896789
5: -1.9334565, 1.7881005, -2.2567866, 1.9983071, -3.9317636, 4.0448871
6: -1.7411426, 1.9812360, -2.0218382, 2.2426498, -3.9837923, 4.0030742
7: -2.0472083, 2.0340009, -2.3672431, 2.3367693, -4.3839779, 4.4012442
8: -2.8049994, 1.3906354, -3.3194478, 1.5786661, -4.3836656, 4.7100830
9: -1.8189635, 1.8345429, -2.1067595, 2.1693935, -3.9883571, 3.9413023

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0650835, 1.7276446, -2.6530287, 2.1486967, -4.2137804, 4.3806734
1: -1.6620727, 1.5962671, -2.1371045, 2.0049031, -3.6669757, 3.7333717
2: -2.0120955, 1.6454037, -2.7328520, 2.0169263, -4.0290217, 4.3782558
3: -2.2770283, 1.3867142, -2.9741049, 1.7251282, -4.0021563, 4.3608189
4: -2.4258995, 1.6516294, -3.1112006, 2.1298561, -4.5557556, 4.7628298
5: -1.9042636, 1.7670783, -2.5278544, 2.1686778, -4.0729413, 4.2949328
6: -1.7162294, 1.9562579, -2.2544136, 2.4562747, -4.1725044, 4.2106714
7: -2.0211225, 2.0081379, -2.6284428, 2.5905178, -4.6116400, 4.6365805
8: -2.7605605, 1.3726192, -3.7032166, 1.7051127, -4.4656734, 5.0758357
9: -1.7949148, 1.8123150, -2.3408747, 2.3662419, -4.1611567, 4.1531897

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.5530794, 2.0842085, -2.5361462, 2.0662789, -4.6193581, 4.6203547
1: -2.0557983, 1.9356909, -2.0492849, 1.9250532, -3.9808517, 3.9849758
2: -2.6061571, 1.9619758, -2.5908852, 1.9513563, -4.5575132, 4.5528612
3: -2.8541253, 1.6687641, -2.8413615, 1.6620383, -4.5161638, 4.5101256
4: -2.9991403, 2.0503845, -2.9868734, 2.0407581, -5.0398984, 5.0372581
5: -2.4226506, 2.1249256, -2.4049091, 2.1016920, -4.5243425, 4.5298347
6: -2.1705265, 2.3895175, -2.1545725, 2.3704898, -4.5410166, 4.5440903
7: -2.5197325, 2.4836340, -2.5102975, 2.4734833, -4.9932156, 4.9939318
8: -3.5676296, 1.6942568, -3.5472908, 1.6748905, -5.2425203, 5.2415476
9: -2.2525322, 2.3232732, -2.2410514, 2.3170941, -4.5696263, 4.5643244

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.5165060, 2.0534053, -2.7867386, 2.2534561, -4.7699623, 4.8401442
1: -2.0274420, 1.9100734, -2.2466912, 2.0991929, -4.1266346, 4.1567645
2: -2.5621052, 1.9374483, -2.8959122, 2.1066322, -4.6687374, 4.8333607
3: -2.8125045, 1.6479367, -3.1337678, 1.8029028, -4.6154075, 4.7817044
4: -2.9573498, 2.0204895, -3.2741139, 2.2414691, -5.1988192, 5.2946033
5: -2.3836212, 2.0948620, -2.6697662, 2.2673578, -4.6509790, 4.7646284
6: -2.1359406, 2.3543887, -2.3820064, 2.5790639, -4.7150044, 4.7363949
7: -2.4840910, 2.4490149, -2.7657773, 2.7217488, -5.2058401, 5.2147923
8: -3.5075102, 1.6673241, -3.9204443, 1.7971760, -5.3046861, 5.5877686
9: -2.2188537, 2.2893434, -2.4696856, 2.5079243, -4.7267780, 4.7590289

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
time: 4.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.71 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668861
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668861
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668866
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2618420, upper bound: 4.2668866
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.1930698, upper bound: 4.2421890
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2649519, upper bound: 4.2646646
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2808662, upper bound: 4.2807667
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2601532, upper bound: 4.2657651
NS_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.1759292, upper bound: 4.1240043
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2606964, upper bound: 4.2614438
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2800680, upper bound: 4.2802284
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2797491, upper bound: 4.2801302
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2657651, upper bound: 4.2601532
NS_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.1240043, upper bound: 4.1759292
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2614438, upper bound: 4.2606964
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2802284, upper bound: 4.2800680
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797491
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2556302, upper bound: 4.2628369
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.71
Output dim: 8, lower bound: -4.2797038, upper bound: 4.2797038

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -1.9366288, 1.6337672, -3.2185571, 3.3428423
1: -1.2903745, 1.2613496, -1.5673178, 1.5009624, -2.7913370, 2.8286674
2: -1.4349308, 1.3510394, -1.8507612, 1.5691757, -3.0041065, 3.2018006
3: -1.7076654, 1.1180336, -2.1284184, 1.3194178, -3.0270834, 3.2464521
4: -1.8715502, 1.2757896, -2.2774007, 1.5524859, -3.4240360, 3.5531902
5: -1.3935781, 1.4331366, -1.7649215, 1.6867763, -3.0803542, 3.1980581
6: -1.2973832, 1.5571899, -1.5985570, 1.8565005, -3.1538837, 3.1557469
7: -1.5392796, 1.5356119, -1.8922317, 1.8724862, -3.4117658, 3.4278436
8: -1.9922013, 1.1882241, -2.5962846, 1.3617575, -3.3539588, 3.7845087
9: -1.3524704, 1.4370781, -1.6760186, 1.7979292, -3.1503997, 3.1130967

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2268199, upper bound: 4.2111177
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2608378, upper bound: 4.2659456
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.8527009, 1.5966368, -1.9366288, 1.6337672, -3.4864683, 3.5332656
1: -1.4868313, 1.4444220, -1.5673178, 1.5009624, -2.9877937, 3.0117397
2: -1.7275662, 1.5125510, -1.8507612, 1.5691757, -3.2967420, 3.3633122
3: -2.0223882, 1.2615709, -2.1284184, 1.3194178, -3.3418059, 3.3899894
4: -2.1773813, 1.4731731, -2.2774007, 1.5524859, -3.7298672, 3.7505739
5: -1.6695801, 1.6089939, -1.7649215, 1.6867763, -3.3563564, 3.3739154
6: -1.5178220, 1.7667497, -1.5985570, 1.8565005, -3.3743224, 3.3653069
7: -1.7920233, 1.7922113, -1.8922317, 1.8724862, -3.6645095, 3.6844430
8: -2.4161141, 1.2641082, -2.5962846, 1.3617575, -3.7778716, 3.8603928
9: -1.5953801, 1.6336577, -1.6760186, 1.7979292, -3.3933091, 3.3096762

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2268199, upper bound: 4.2111177
time: 2.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2608378, upper bound: 4.2659456
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -2.1927042, 1.8270174, -3.4118073, 3.5989177
1: -1.2903745, 1.2613496, -1.7667007, 1.6783012, -2.9686756, 3.0280504
2: -1.4349308, 1.3510394, -2.1604946, 1.7222885, -3.1572194, 3.5115340
3: -1.7076654, 1.1180336, -2.4245210, 1.4634503, -3.1711159, 3.5425546
4: -1.8715502, 1.2757896, -2.5693102, 1.7527267, -3.6242769, 3.8450999
5: -1.3935781, 1.4331366, -2.0367942, 1.8529298, -3.2465079, 3.4699306
6: -1.2973832, 1.5571899, -1.8287529, 2.0667644, -3.3641477, 3.3859429
7: -1.5392796, 1.5356119, -2.1492403, 2.1234856, -3.6627650, 3.6848521
8: -1.9922013, 1.1882241, -2.9839160, 1.4802775, -3.4724789, 4.1721401
9: -1.3524704, 1.4370781, -1.9085587, 1.9913862, -3.3438566, 3.3456368

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2216081, upper bound: 4.2017555
time: 2.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2606533, upper bound: 4.2659112
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8525033, 1.5966368, -2.1927042, 1.8270174, -3.6795206, 3.7893410
1: -1.4860272, 1.4444220, -1.7667007, 1.6783012, -3.1643286, 3.2111228
2: -1.7269343, 1.5125510, -2.1604946, 1.7222885, -3.4492228, 3.6730456
3: -2.0220840, 1.2615709, -2.4245210, 1.4634503, -3.4855342, 3.6860919
4: -2.1773813, 1.4720203, -2.5693102, 1.7527267, -3.9301081, 4.0413303
5: -1.6695801, 1.6086042, -2.0367942, 1.8529298, -3.5225101, 3.6453984
6: -1.5174130, 1.7667497, -1.8287529, 2.0667644, -3.5841775, 3.5955026
7: -1.7913712, 1.7922113, -2.1492403, 2.1234856, -3.9148569, 3.9414515
8: -2.4161141, 1.2635756, -2.9839160, 1.4802775, -3.8963916, 4.2474918
9: -1.5953801, 1.6335883, -1.9085587, 1.9913862, -3.5867662, 3.5421472

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2216081, upper bound: 4.2017555
time: 2.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2606533, upper bound: 4.2659112
time: 2.92 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.3413130, 1.2534575, -2.0307040, 1.7298583, -3.0711713, 3.2841616
1: -1.1064329, 1.0837612, -1.6464386, 1.5650725, -2.6715055, 2.7301998
2: -1.1643872, 1.2077436, -1.9610318, 1.6288445, -2.7932317, 3.1687756
3: -1.4114528, 0.9982265, -2.2394998, 1.3742992, -2.7857518, 3.2377262
4: -1.5771528, 1.1021525, -2.3906744, 1.6232077, -3.2003605, 3.4928269
5: -1.1708810, 1.2549249, -1.8666571, 1.7631229, -2.9340038, 3.1215820
6: -1.0917606, 1.3843800, -1.6859518, 1.9550505, -3.0468111, 3.0703318
7: -1.3048267, 1.3104151, -1.9817816, 1.9579840, -3.2628107, 3.2921968
8: -1.6211197, 1.2121111, -2.7554388, 1.4362445, -3.0573642, 3.9675498
9: -1.1512780, 1.3249016, -1.7619985, 1.9154980, -3.0667760, 3.0869002

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1699042, upper bound: 4.0396835
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.1699042, upper bound: 4.2421890
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.9482822, 1.6480703, -2.2065530, 1.8596144, -3.8078966, 3.8546233
1: -1.5734551, 1.5089953, -1.7857372, 1.6896710, -3.2631261, 3.2947326
2: -1.8592560, 1.5786078, -2.1739576, 1.7406670, -3.5999231, 3.7525654
3: -2.1378491, 1.3252633, -2.4433260, 1.4767292, -3.6145782, 3.7685893
4: -2.2920995, 1.5581020, -2.5955205, 1.7650933, -4.0571928, 4.1536226
5: -1.7776377, 1.7083184, -2.0571856, 1.8935564, -3.6711941, 3.7655039
6: -1.6119969, 1.8784479, -1.8475021, 2.1045749, -3.7165718, 3.7259500
7: -1.8976521, 1.8785890, -2.1588426, 2.1302233, -4.0278754, 4.0374317
8: -2.6199908, 1.3914015, -3.0295806, 1.5362818, -4.1562729, 4.4209824
9: -1.6874927, 1.8279969, -1.9258299, 2.0635457, -3.7510386, 3.7538266

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2439017, upper bound: 4.0462149
time: 1.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2439017, upper bound: 4.2646646
time: 1.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.1039996, 1.7637614, -2.0704732, 1.7321072, -3.8361068, 3.8342347
1: -1.6962105, 1.6183649, -1.6755357, 1.5946895, -3.2909000, 3.2939005
2: -2.0477369, 1.6768968, -2.0115166, 1.6540654, -3.7018023, 3.6884134
3: -2.3186967, 1.4140369, -2.2864597, 1.3964233, -3.7151201, 3.7004967
4: -2.4741194, 1.6818365, -2.4398639, 1.6581353, -4.1322546, 4.1217003
5: -1.9457788, 1.8224949, -1.9070085, 1.7859906, -3.7317696, 3.7295034
6: -1.7551769, 2.0107517, -1.7230678, 1.9743704, -3.7295473, 3.7338195
7: -2.0534024, 2.0305119, -2.0263715, 2.0018229, -4.0552254, 4.0568833
8: -2.8671131, 1.4764357, -2.8176849, 1.4459844, -4.3130975, 4.2941208
9: -1.8330430, 1.9621764, -1.8035028, 1.9407465, -3.7737894, 3.7656794

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
time: 3.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0689967, 1.7350297, -2.3249154, 1.9275001, -3.9964967, 4.0599451
1: -1.6691093, 1.5937312, -1.8744260, 1.7717481, -3.4408574, 3.4681573
2: -2.0059993, 1.6534979, -2.3204210, 1.8091429, -3.8151422, 3.9739189
3: -2.2789984, 1.3939416, -2.5819590, 1.5393851, -3.8183835, 3.9759007
4: -2.4336066, 1.6537644, -2.7309556, 1.8606935, -4.2943001, 4.3847198
5: -1.9075590, 1.7934947, -2.1773515, 1.9514437, -3.8590026, 3.9708462
6: -1.7220417, 1.9777493, -1.9532542, 2.1863575, -3.9083991, 3.9310036
7: -2.0196769, 1.9972435, -2.2835886, 2.2522337, -4.2719107, 4.2808323
8: -2.8083694, 1.4517686, -3.2014475, 1.5677909, -4.3761601, 4.6532164
9: -1.8003739, 1.9297739, -2.0347838, 2.1323524, -3.9327264, 3.9645576

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
time: 1.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
time: 2.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -2.3968477, 1.9568636, -3.5416536, 3.8030612
1: -1.2903745, 1.2613496, -1.9351065, 1.8265558, -3.1169305, 3.1964560
2: -1.4349308, 1.3510394, -2.4212003, 1.8581847, -3.2931156, 3.7722397
3: -1.7076654, 1.1180336, -2.6757462, 1.5809891, -3.2886546, 3.7937799
4: -1.8715502, 1.2757896, -2.8170159, 1.9246368, -3.7961869, 4.0928054
5: -1.3935781, 1.4331366, -2.2567866, 1.9988856, -3.3924637, 3.6899233
6: -1.2973832, 1.5571899, -2.0223429, 2.2426498, -3.5400329, 3.5795329
7: -1.5392796, 1.5356119, -2.3676341, 2.3367693, -3.8760490, 3.9032459
8: -1.9922013, 1.1882241, -3.3195601, 1.5786661, -3.5708675, 4.5077839
9: -1.3524704, 1.4370781, -2.1067595, 2.1695790, -3.5220494, 3.5438375

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1775301, upper bound: 4.1726229
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2591637, upper bound: 4.2648364
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.8527009, 1.5966368, -2.3968477, 1.9568636, -3.8095646, 3.9934845
1: -1.4868313, 1.4444220, -1.9351065, 1.8265558, -3.3133872, 3.3795285
2: -1.7275662, 1.5125510, -2.4212003, 1.8581847, -3.5857511, 3.9337511
3: -2.0223882, 1.2615709, -2.6757462, 1.5809891, -3.6033773, 3.9373171
4: -2.1773813, 1.4731731, -2.8170159, 1.9246368, -4.1020184, 4.2901888
5: -1.6695801, 1.6089939, -2.2567866, 1.9988856, -3.6684656, 3.8657804
6: -1.5178220, 1.7667497, -2.0223429, 2.2426498, -3.7604718, 3.7890925
7: -1.7920233, 1.7922113, -2.3676341, 2.3367693, -4.1287928, 4.1598454
8: -2.4161141, 1.2641082, -3.3195601, 1.5786661, -3.9947801, 4.5836682
9: -1.5953801, 1.6336577, -2.1067595, 2.1695790, -3.7649591, 3.7404172

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1775301, upper bound: 4.1726229
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2591637, upper bound: 4.2648364
time: 3.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -2.6570215, 2.1486967, -3.7334867, 4.0632353
1: -1.2903745, 1.2613496, -2.1382222, 2.0060167, -3.2963912, 3.3995719
2: -1.4349308, 1.3510394, -2.7360160, 2.0174913, -3.4524221, 4.0870552
3: -1.7076654, 1.1180336, -2.9781241, 1.7255940, -3.4332595, 4.0961580
4: -1.8715502, 1.2757896, -3.1112006, 2.1333952, -4.0049453, 4.3869901
5: -1.3935781, 1.4331366, -2.5278871, 2.1722200, -3.5657980, 3.9610238
6: -1.2973832, 1.5571899, -2.2574952, 2.4575932, -3.7549763, 3.8146851
7: -1.5392796, 1.5356119, -2.6308317, 2.5906546, -4.1299343, 4.1664438
8: -1.9922013, 1.1882241, -3.7043788, 1.7051127, -3.6973140, 4.8926029
9: -1.3524704, 1.4370781, -2.3408747, 2.3680873, -3.7205577, 3.7779527

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1579011, upper bound: 4.1553247
time: 2.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2588208, upper bound: 4.2647055
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8525033, 1.5966368, -2.6570215, 2.1486967, -4.0011997, 4.2536583
1: -1.4860272, 1.4444220, -2.1382222, 2.0060167, -3.4920440, 3.5826442
2: -1.7269343, 1.5125510, -2.7360160, 2.0174913, -3.7444258, 4.2485671
3: -2.0220840, 1.2615709, -2.9781241, 1.7255940, -3.7476780, 4.2396951
4: -2.1773813, 1.4720203, -3.1112006, 2.1333952, -4.3107767, 4.5832210
5: -1.6695801, 1.6086042, -2.5278871, 2.1722200, -3.8418002, 4.1364913
6: -1.5174130, 1.7667497, -2.2574952, 2.4575932, -3.9750061, 4.0242448
7: -1.7913712, 1.7922113, -2.6308317, 2.5906546, -4.3820257, 4.4230433
8: -2.4161141, 1.2635756, -3.7043788, 1.7051127, -4.1212268, 4.9679546
9: -1.5953801, 1.6335883, -2.3408747, 2.3680873, -3.9634674, 3.9744630

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1579011, upper bound: 4.1553247
time: 2.04 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2588208, upper bound: 4.2647055
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0914512, 1.7711089, -2.5239367, 2.0639822, -4.1554337, 4.2950459
1: -1.6925064, 1.6076061, -2.0342703, 1.9155869, -3.6080933, 3.6418762
2: -2.0345435, 1.6656239, -2.5708663, 1.9446722, -3.9792156, 4.2364902
3: -2.3086519, 1.4077268, -2.8215320, 1.6531169, -3.9617689, 4.2292585
4: -2.4584424, 1.6701031, -2.9684014, 2.0280619, -4.4865046, 4.6385045
5: -1.9332569, 1.8058885, -2.3911114, 2.1052568, -4.0385137, 4.1970000
6: -1.7391957, 2.0005102, -2.1455412, 2.3684237, -4.1076193, 4.1460514
7: -2.0426583, 2.0184693, -2.4907839, 2.4548697, -4.4975281, 4.5092535
8: -2.8418233, 1.4601251, -3.5286887, 1.6835133, -4.5253367, 4.9888139
9: -1.8173971, 1.9517206, -2.2270379, 2.3100853, -4.1274824, 4.1787586

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0285212, upper bound: 4.2307242
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0285212, upper bound: 4.2614438
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.0913379, 1.7541999, -2.5385122, 2.0662789, -4.1576166, 4.2927122
1: -1.6863245, 1.6094422, -2.0499606, 1.9254878, -3.6118121, 3.6594028
2: -2.0324769, 1.6687468, -2.5928035, 1.9515772, -3.9840541, 4.2615504
3: -2.3040586, 1.4067905, -2.8437948, 1.6622201, -3.9662786, 4.2505856
4: -2.4592841, 1.6715741, -2.9868734, 2.0424118, -4.5016956, 4.6584473
5: -1.9322655, 1.8132346, -2.4049220, 2.1038218, -4.0360870, 4.2181568
6: -1.7432476, 1.9995810, -2.1564331, 2.3710079, -4.1142554, 4.1560140
7: -2.0408397, 2.0181684, -2.5117409, 2.4735363, -4.5143757, 4.5299091
8: -2.8465054, 1.4688146, -3.5478897, 1.6748905, -4.5213957, 5.0167046
9: -1.8212051, 1.9510278, -2.2410514, 2.3180361, -4.1392412, 4.1920791

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
time: 2.92 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.0564189, 1.7256352, -2.7924316, 2.2534561, -4.3098750, 4.5180669
1: -1.6593251, 1.5848820, -2.2482793, 2.1008081, -3.7601333, 3.8331614
2: -1.9909006, 1.6454715, -2.9004233, 2.1074500, -4.0983505, 4.5458946
3: -2.2644575, 1.3867835, -3.1394947, 1.8035789, -4.0680361, 4.5262780
4: -2.4188945, 1.6436236, -3.2741139, 2.2465856, -4.6654801, 4.9177375
5: -1.8941505, 1.7843351, -2.6698141, 2.2723768, -4.1665273, 4.4541492
6: -1.7102883, 1.9666922, -2.3863821, 2.5809729, -4.2912612, 4.3530741
7: -2.0071874, 1.9850165, -2.7691746, 2.7219460, -4.7291336, 4.7541909
8: -2.7882311, 1.4443891, -3.9220977, 1.7971760, -4.5854073, 5.3664865
9: -1.7886047, 1.9188341, -2.4696856, 2.5105207, -4.2991257, 4.3885198

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
time: 2.11 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.3968477, 1.9568636, -1.5847899, 1.4062135, -3.8030612, 3.5416536
1: -1.9351065, 1.8265558, -1.2903745, 1.2613496, -3.1964560, 3.1169305
2: -2.4212003, 1.8581847, -1.4349308, 1.3510394, -3.7722397, 3.2931156
3: -2.6757462, 1.5809891, -1.7076654, 1.1180336, -3.7937799, 3.2886546
4: -2.8170159, 1.9246368, -1.8715502, 1.2757896, -4.0928054, 3.7961869
5: -2.2567866, 1.9988856, -1.3935781, 1.4331366, -3.6899233, 3.3924637
6: -2.0223429, 2.2426498, -1.2973832, 1.5571899, -3.5795329, 3.5400329
7: -2.3676341, 2.3367693, -1.5392796, 1.5356119, -3.9032459, 3.8760490
8: -3.3195601, 1.5786661, -1.9922013, 1.1882241, -4.5077839, 3.5708675
9: -2.1067595, 2.1695790, -1.3524704, 1.4370781, -3.5438375, 3.5220494

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1726229, upper bound: 4.1775301
time: 2.03 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2648364, upper bound: 4.2591637
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.3968477, 1.9568636, -1.8527009, 1.5966368, -3.9934845, 3.8095646
1: -1.9351065, 1.8265558, -1.4868313, 1.4444220, -3.3795285, 3.3133872
2: -2.4212003, 1.8581847, -1.7275662, 1.5125510, -3.9337511, 3.5857511
3: -2.6757462, 1.5809891, -2.0223882, 1.2615709, -3.9373171, 3.6033773
4: -2.8170159, 1.9246368, -2.1773813, 1.4731731, -4.2901888, 4.1020184
5: -2.2567866, 1.9988856, -1.6695801, 1.6089939, -3.8657804, 3.6684656
6: -2.0223429, 2.2426498, -1.5178220, 1.7667497, -3.7890925, 3.7604718
7: -2.3676341, 2.3367693, -1.7920233, 1.7922113, -4.1598454, 4.1287928
8: -3.3195601, 1.5786661, -2.4161141, 1.2641082, -4.5836682, 3.9947801
9: -2.1067595, 2.1695790, -1.5953801, 1.6336577, -3.7404172, 3.7649591

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1726229, upper bound: 4.1775301
time: 2.07 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2648364, upper bound: 4.2591637
time: 1.70 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.6570215, 2.1486967, -1.5847899, 1.4062135, -4.0632353, 3.7334867
1: -2.1382222, 2.0060167, -1.2903745, 1.2613496, -3.3995719, 3.2963912
2: -2.7360160, 2.0174913, -1.4349308, 1.3510394, -4.0870552, 3.4524221
3: -2.9781241, 1.7255940, -1.7076654, 1.1180336, -4.0961580, 3.4332595
4: -3.1112006, 2.1333952, -1.8715502, 1.2757896, -4.3869901, 4.0049453
5: -2.5278871, 2.1722200, -1.3935781, 1.4331366, -3.9610238, 3.5657980
6: -2.2574952, 2.4575932, -1.2973832, 1.5571899, -3.8146851, 3.7549763
7: -2.6308317, 2.5906546, -1.5392796, 1.5356119, -4.1664438, 4.1299343
8: -3.7043788, 1.7051127, -1.9922013, 1.1882241, -4.8926029, 3.6973140
9: -2.3408747, 2.3680873, -1.3524704, 1.4370781, -3.7779527, 3.7205577

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1553247, upper bound: 4.1579011
time: 3.00 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2647055, upper bound: 4.2588208
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.6570215, 2.1486967, -1.8525033, 1.5966368, -4.2536583, 4.0011997
1: -2.1382222, 2.0060167, -1.4860272, 1.4444220, -3.5826442, 3.4920440
2: -2.7360160, 2.0174913, -1.7269343, 1.5125510, -4.2485671, 3.7444258
3: -2.9781241, 1.7255940, -2.0220840, 1.2615709, -4.2396951, 3.7476780
4: -3.1112006, 2.1333952, -2.1773813, 1.4720203, -4.5832210, 4.3107767
5: -2.5278871, 2.1722200, -1.6695801, 1.6086042, -4.1364913, 3.8418002
6: -2.2574952, 2.4575932, -1.5174130, 1.7667497, -4.0242448, 3.9750061
7: -2.6308317, 2.5906546, -1.7913712, 1.7922113, -4.4230433, 4.3820257
8: -3.7043788, 1.7051127, -2.4161141, 1.2635756, -4.9679546, 4.1212268
9: -2.3408747, 2.3680873, -1.5953801, 1.6335883, -3.9744630, 3.9634674

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1553247, upper bound: 4.1579011
time: 1.85 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2647055, upper bound: 4.2588208
time: 2.53 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.5239367, 2.0639822, -2.0914512, 1.7711089, -4.2950459, 4.1554337
1: -2.0342703, 1.9155869, -1.6925064, 1.6076061, -3.6418762, 3.6080933
2: -2.5708663, 1.9446722, -2.0345435, 1.6656239, -4.2364902, 3.9792156
3: -2.8215320, 1.6531169, -2.3086519, 1.4077268, -4.2292585, 3.9617689
4: -2.9684014, 2.0280619, -2.4584424, 1.6701031, -4.6385045, 4.4865046
5: -2.3911114, 2.1052568, -1.9332569, 1.8058885, -4.1970000, 4.0385137
6: -2.1455412, 2.3684237, -1.7391957, 2.0005102, -4.1460514, 4.1076193
7: -2.4907839, 2.4548697, -2.0426583, 2.0184693, -4.5092535, 4.4975281
8: -3.5286887, 1.6835133, -2.8418233, 1.4601251, -4.9888139, 4.5253367
9: -2.2270379, 2.3100853, -1.8173971, 1.9517206, -4.1787586, 4.1274824

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2307242, upper bound: 4.0285212
time: 1.94 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2307242, upper bound: 4.0285212
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2.5385122, 2.0662789, -2.0913379, 1.7541999, -4.2927122, 4.1576166
1: -2.0499606, 1.9254878, -1.6863245, 1.6094422, -3.6594028, 3.6118121
2: -2.5928035, 1.9515772, -2.0324769, 1.6687468, -4.2615504, 3.9840541
3: -2.8437948, 1.6622201, -2.3040586, 1.4067905, -4.2505856, 3.9662786
4: -2.9868734, 2.0424118, -2.4592841, 1.6715741, -4.6584473, 4.5016956
5: -2.4049220, 2.1038218, -1.9322655, 1.8132346, -4.2181568, 4.0360870
6: -2.1564331, 2.3710079, -1.7432476, 1.9995810, -4.1560140, 4.1142554
7: -2.5117409, 2.4735363, -2.0408397, 2.0181684, -4.5299091, 4.5143757
8: -3.5478897, 1.6748905, -2.8465054, 1.4688146, -5.0167046, 4.5213957
9: -2.2410514, 2.3180361, -1.8212051, 1.9510278, -4.1920791, 4.1392412

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797487
time: 1.78 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797491
time: 2.48 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.7924316, 2.2534561, -2.0564189, 1.7256352, -4.5180669, 4.3098750
1: -2.2482793, 2.1008081, -1.6593251, 1.5848820, -3.8331614, 3.7601333
2: -2.9004233, 2.1074500, -1.9909006, 1.6454715, -4.5458946, 4.0983505
3: -3.1394947, 1.8035789, -2.2644575, 1.3867835, -4.5262780, 4.0680361
4: -3.2741139, 2.2465856, -2.4188945, 1.6436236, -4.9177375, 4.6654801
5: -2.6698141, 2.2723768, -1.8941505, 1.7843351, -4.4541492, 4.1665273
6: -2.3863821, 2.5809729, -1.7102883, 1.9666922, -4.3530741, 4.2912612
7: -2.7691746, 2.7219460, -2.0071874, 1.9850165, -4.7541909, 4.7291336
8: -3.9220977, 1.7971760, -2.7882311, 1.4443891, -5.3664865, 4.5854073
9: -2.4696856, 2.5105207, -1.7886047, 1.9188341, -4.3885198, 4.2991257

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797487
time: 4.40 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797491
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.9950472, 1.6754459, -2.3962343, 1.9568636, -3.9519110, 4.0716801
1: -1.6089311, 1.5494200, -1.9349238, 1.8265558, -3.4354868, 3.4843438
2: -1.9290595, 1.6034020, -2.4206822, 1.8581847, -3.7872443, 4.0240841
3: -2.1982839, 1.3457385, -2.6750891, 1.5809891, -3.7792730, 4.0208278
4: -2.3512256, 1.5964495, -2.8170159, 1.9244330, -4.2756586, 4.4134655
5: -1.8267596, 1.7097182, -2.2567866, 1.9983071, -3.8250666, 3.9665048
6: -1.6511068, 1.8918917, -2.0218382, 2.2426498, -3.8937566, 3.9137299
7: -1.9541341, 1.9398717, -2.3672431, 2.3367693, -4.2909036, 4.3071146
8: -2.6519578, 1.3297880, -3.3194478, 1.5786661, -4.2306237, 4.6492357
9: -1.7316319, 1.7575376, -2.1067595, 2.1693935, -3.9010253, 3.8642972

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1294841, upper bound: 4.1074897
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2498426, upper bound: 4.2615243
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.2629788, 1.8733602, -2.3962343, 1.9568636, -4.2198424, 4.2695942
1: -1.8168786, 1.7348837, -1.9349238, 1.8265558, -3.6434345, 3.6698074
2: -2.2549591, 1.7643305, -2.4206822, 1.8581847, -4.1131439, 4.1850128
3: -2.5105658, 1.4987251, -2.6750891, 1.5809891, -4.0915546, 4.1738143
4: -2.6577535, 1.8086014, -2.8170159, 1.9244330, -4.5821867, 4.6256170
5: -2.1136882, 1.8859401, -2.2567866, 1.9983071, -4.1119952, 4.1427269
6: -1.8938354, 2.1081598, -2.0218382, 2.2426498, -4.1364851, 4.1299982
7: -2.2260754, 2.2064371, -2.3672431, 2.3367693, -4.5628448, 4.5736799
8: -3.0693290, 1.4530313, -3.3194478, 1.5786661, -4.6479950, 4.7724791
9: -1.9789912, 1.9680797, -2.1067595, 2.1693935, -4.1483846, 4.0748391

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1294841, upper bound: 4.1074897
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2498426, upper bound: 4.2615243
time: 2.05 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.9950472, 1.6754459, -2.6530287, 2.1486967, -4.1437440, 4.3284745
1: -1.6089311, 1.5494200, -2.1371045, 2.0049031, -3.6138341, 3.6865244
2: -1.9290595, 1.6034020, -2.7328520, 2.0169263, -3.9459858, 4.3362541
3: -2.1982839, 1.3457385, -2.9741049, 1.7251282, -3.9234121, 4.3198433
4: -2.3512256, 1.5964495, -3.1112006, 2.1298561, -4.4810820, 4.7076502
5: -1.8267596, 1.7097182, -2.5278544, 2.1686778, -3.9954374, 4.2375727
6: -1.6511068, 1.8918917, -2.2544136, 2.4562747, -4.1073818, 4.1463051
7: -1.9541341, 1.9398717, -2.6284428, 2.5905178, -4.5446520, 4.5683146
8: -2.6519578, 1.3297880, -3.7032166, 1.7051127, -4.3570704, 5.0330048
9: -1.7316319, 1.7575376, -2.3408747, 2.3662419, -4.0978737, 4.0984125

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1096914, upper bound: 4.0961760
time: 3.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2496300, upper bound: 4.2614978
time: 5.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.2629788, 1.8733602, -2.6530287, 2.1486967, -4.4116755, 4.5263891
1: -1.8168786, 1.7348837, -2.1371045, 2.0049031, -3.8217816, 3.8719883
2: -2.2549591, 1.7643305, -2.7328520, 2.0169263, -4.2718854, 4.4971824
3: -2.5105658, 1.4987251, -2.9741049, 1.7251282, -4.2356939, 4.4728298
4: -2.6577535, 1.8086014, -3.1112006, 2.1298561, -4.7876096, 4.9198017
5: -2.1136882, 1.8859401, -2.5278544, 2.1686778, -4.2823658, 4.4137945
6: -1.8938354, 2.1081598, -2.2544136, 2.4562747, -4.3501101, 4.3625736
7: -2.2260754, 2.2064371, -2.6284428, 2.5905178, -4.8165932, 4.8348799
8: -3.0693290, 1.4530313, -3.7032166, 1.7051127, -4.7744417, 5.1562481
9: -1.9789912, 1.9680797, -2.3408747, 2.3662419, -4.3452330, 4.3089542

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1096914, upper bound: 4.0961760
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2496300, upper bound: 4.2614978
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.4181757, 1.9738275, -2.5361462, 2.0662789, -4.4844546, 4.5099735
1: -1.9532030, 1.8411082, -2.0492849, 1.9250532, -3.8782563, 3.8903933
2: -2.4461794, 1.8727127, -2.5908852, 1.9513563, -4.3975358, 4.4635978
3: -2.7017128, 1.5930214, -2.8413615, 1.6620383, -4.3637514, 4.4343829
4: -2.8450613, 1.9420012, -2.9868734, 2.0407581, -4.8858194, 4.9288745
5: -2.2784529, 2.0144191, -2.4049091, 2.1016920, -4.3801451, 4.4193282
6: -2.0428829, 2.2621255, -2.1545725, 2.3704898, -4.4133730, 4.4166980
7: -2.3895493, 2.3568246, -2.5102975, 2.4734833, -4.8630323, 4.8671222
8: -3.3547430, 1.5962609, -3.5472908, 1.6748905, -5.0296335, 5.1435518
9: -2.1284590, 2.2009857, -2.2410514, 2.3170941, -4.4455528, 4.4420371

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2655664, upper bound: 4.2611386
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796425, upper bound: 4.2794701
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.6759152, 2.1643093, -2.5361462, 2.0662789, -4.7421942, 4.7004557
1: -2.1544385, 2.0189829, -2.0492849, 1.9250532, -4.0794916, 4.0682678
2: -2.7579510, 2.0307910, -2.5908852, 1.9513563, -4.7093072, 4.6216764
3: -3.0013902, 1.7363446, -2.8413615, 1.6620383, -4.6634283, 4.5777063
4: -3.1366315, 2.1488299, -2.9868734, 2.0407581, -5.1773896, 5.1357031
5: -2.5472465, 2.1865449, -2.4049091, 2.1016920, -4.6489382, 4.5914540
6: -2.2760525, 2.4756329, -2.1545725, 2.3704898, -4.6465425, 4.6302052
7: -2.6503747, 2.6084881, -2.5102975, 2.4734833, -5.1238580, 5.1187859
8: -3.7347119, 1.7219852, -3.5472908, 1.6748905, -5.4096022, 5.2692761
9: -2.3602483, 2.3979821, -2.2410514, 2.3170941, -4.6773424, 4.6390333

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2655664, upper bound: 4.2611386
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796425, upper bound: 4.2794701
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.4181757, 1.9738275, -2.7867386, 2.2534561, -4.6716318, 4.7605662
1: -1.9532030, 1.8411082, -2.2466912, 2.0991929, -4.0523958, 4.0877995
2: -2.4461794, 1.8727127, -2.8959122, 2.1066322, -4.5528116, 4.7686248
3: -2.7017128, 1.5930214, -3.1337678, 1.8029028, -4.5046158, 4.7267895
4: -2.8450613, 1.9420012, -3.2741139, 2.2414691, -5.0865307, 5.2161150
5: -2.2784529, 2.0144191, -2.6697662, 2.2673578, -4.5458107, 4.6841850
6: -2.0428829, 2.2621255, -2.3820064, 2.5790639, -4.6219468, 4.6441317
7: -2.3895493, 2.3568246, -2.7657773, 2.7217488, -5.1112981, 5.1226020
8: -3.3547430, 1.5962609, -3.9204443, 1.7971760, -5.1519189, 5.5167050
9: -2.1284590, 2.2009857, -2.4696856, 2.5079243, -4.6363831, 4.6706715

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2539438, upper bound: 4.2641668
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793658, upper bound: 4.2793658
time: 2.04 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.6759152, 2.1643093, -2.7867386, 2.2534561, -4.9293714, 4.9510479
1: -2.1544385, 2.0189829, -2.2466912, 2.0991929, -4.2536316, 4.2656741
2: -2.7579510, 2.0307910, -2.8959122, 2.1066322, -4.8645830, 4.9267035
3: -3.0013902, 1.7363446, -3.1337678, 1.8029028, -4.8042932, 4.8701124
4: -3.1366315, 2.1488299, -3.2741139, 2.2414691, -5.3781004, 5.4229441
5: -2.5472465, 2.1865449, -2.6697662, 2.2673578, -4.8146043, 4.8563108
6: -2.2760525, 2.4756329, -2.3820064, 2.5790639, -4.8551164, 4.8576393
7: -2.6503747, 2.6084881, -2.7657773, 2.7217488, -5.3721237, 5.3742657
8: -3.7347119, 1.7219852, -3.9204443, 1.7971760, -5.5318880, 5.6424294
9: -2.3602483, 2.3979821, -2.4696856, 2.5079243, -4.8681726, 4.8676677

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2628369, upper bound: 4.2556302
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2628369, upper bound: 4.2795244
time: 3.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 9.32 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2268199, upper bound: 4.2111177
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2608378, upper bound: 4.2659456
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2268199, upper bound: 4.2111177
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2608378, upper bound: 4.2659456
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2216081, upper bound: 4.2017555
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2606533, upper bound: 4.2659112
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2216081, upper bound: 4.2017555
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2606533, upper bound: 4.2659112
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1699042, upper bound: 4.0396835
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1699042, upper bound: 4.2421890
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2439017, upper bound: 4.0462149
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2439017, upper bound: 4.2646646
NS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
NS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
NS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2806844, upper bound: 4.2806844
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1775301, upper bound: 4.1726229
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2591637, upper bound: 4.2648364
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1775301, upper bound: 4.1726229
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2591637, upper bound: 4.2648364
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1579011, upper bound: 4.1553247
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2588208, upper bound: 4.2647055
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1579011, upper bound: 4.1553247
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2588208, upper bound: 4.2647055
NS_A1_B2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.0285212, upper bound: 4.2307242
NS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.0285212, upper bound: 4.2614438
NS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
NS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
NS_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
NS_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2797487, upper bound: 4.2801302
NS_A2_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1726229, upper bound: 4.1775301
NS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2648364, upper bound: 4.2591637
NS_A2_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1726229, upper bound: 4.1775301
NS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2648364, upper bound: 4.2591637
NS_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1553247, upper bound: 4.1579011
NS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2647055, upper bound: 4.2588208
NS_A2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1553247, upper bound: 4.1579011
NS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2647055, upper bound: 4.2588208
NS_A2_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2307242, upper bound: 4.0285212
NS_A2_B1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2307242, upper bound: 4.0285212
NS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797487
NS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797491
NS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797487
NS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2801302, upper bound: 4.2797491
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1294841, upper bound: 4.1074897
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2498426, upper bound: 4.2615243
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1294841, upper bound: 4.1074897
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2498426, upper bound: 4.2615243
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1096914, upper bound: 4.0961760
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2496300, upper bound: 4.2614978
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.1096914, upper bound: 4.0961760
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2496300, upper bound: 4.2614978
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2655664, upper bound: 4.2611386
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2796425, upper bound: 4.2794701
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2655664, upper bound: 4.2611386
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2796425, upper bound: 4.2794701
NS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2539438, upper bound: 4.2641668
NS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2793658, upper bound: 4.2793658
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2628369, upper bound: 4.2556302
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.32
Output dim: 8, lower bound: -4.2628369, upper bound: 4.2795244

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -1.8164266, 1.5494646, -3.1342545, 3.2226400
1: -1.2903745, 1.2613496, -1.4732368, 1.4193708, -2.7097454, 2.7345862
2: -1.4349308, 1.3510394, -1.7052919, 1.4958303, -2.9307611, 3.0563312
3: -1.7076654, 1.1180336, -1.9888200, 1.2505196, -2.9581852, 3.1068535
4: -1.8715502, 1.2757896, -2.1415181, 1.4580431, -3.3295932, 3.4173079
5: -1.3935781, 1.4331366, -1.6354316, 1.6011187, -2.9946966, 3.0685682
6: -1.2973832, 1.5571899, -1.4934781, 1.7578225, -3.0552058, 3.0506680
7: -1.5392796, 1.5356119, -1.7715780, 1.7551513, -3.2944307, 3.3071899
8: -1.9922013, 1.1882241, -2.4038806, 1.3052739, -3.2974753, 3.5921047
9: -1.3524704, 1.4370781, -1.5647489, 1.6953074, -3.0477777, 3.0018270

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0520408, upper bound: 4.2461377
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0520408, upper bound: 4.2677839
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.8527009, 1.5966368, -1.8164266, 1.5494646, -3.4021654, 3.4130635
1: -1.4868313, 1.4444220, -1.4732368, 1.4193708, -2.9062021, 2.9176588
2: -1.7275662, 1.5125510, -1.7052919, 1.4958303, -3.2233965, 3.2178428
3: -2.0223882, 1.2615709, -1.9888200, 1.2505196, -3.2729077, 3.2503910
4: -2.1773813, 1.4731731, -2.1415181, 1.4580431, -3.6354244, 3.6146913
5: -1.6695801, 1.6089939, -1.6354316, 1.6011187, -3.2706988, 3.2444255
6: -1.5178220, 1.7667497, -1.4934781, 1.7578225, -3.2756445, 3.2602277
7: -1.7920233, 1.7922113, -1.7715780, 1.7551513, -3.5471745, 3.5637894
8: -2.4161141, 1.2641082, -2.4038806, 1.3052739, -3.7213879, 3.6679888
9: -1.5953801, 1.6336577, -1.5647489, 1.6953074, -3.2906876, 3.1984067

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.9875687, upper bound: 4.2416380
time: 4.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.9875687, upper bound: 4.2659457
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -2.0702529, 1.7414235, -3.3262134, 3.4764664
1: -1.2903745, 1.2613496, -1.6693184, 1.5926604, -2.8830349, 2.9306679
2: -1.4349308, 1.3510394, -2.0115924, 1.6470380, -3.0819688, 3.3626318
3: -1.7076654, 1.1180336, -2.2814622, 1.3934726, -3.1011381, 3.3994958
4: -1.8715502, 1.2757896, -2.4285827, 1.6557410, -3.5272913, 3.7043724
5: -1.3935781, 1.4331366, -1.9048606, 1.7657413, -3.1593194, 3.3379972
6: -1.2973832, 1.5571899, -1.7179687, 1.9643700, -3.2617531, 3.2751586
7: -1.5392796, 1.5356119, -2.0255573, 2.0026951, -3.5419745, 3.5611691
8: -1.9922013, 1.1882241, -2.7902472, 1.4153975, -3.4075990, 3.9784713
9: -1.3524704, 1.4370781, -1.7957036, 1.8860987, -3.2385693, 3.2327819

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0482152, upper bound: 4.2448266
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0482152, upper bound: 4.2670566
time: 6.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8525033, 1.5966368, -2.0702529, 1.7414235, -3.5939269, 3.6668897
1: -1.4860272, 1.4444220, -1.6693184, 1.5926604, -3.0786877, 3.1137404
2: -1.7269343, 1.5125510, -2.0115924, 1.6470380, -3.3739724, 3.5241432
3: -2.0220840, 1.2615709, -2.2814622, 1.3934726, -3.4155564, 3.5430331
4: -2.1773813, 1.4720203, -2.4285827, 1.6557410, -3.8331223, 3.9006028
5: -1.6695801, 1.6086042, -1.9048606, 1.7657413, -3.4353213, 3.5134649
6: -1.5174130, 1.7667497, -1.7179687, 1.9643700, -3.4817829, 3.4847183
7: -1.7913712, 1.7922113, -2.0255573, 2.0026951, -3.7940664, 3.8177686
8: -2.4161141, 1.2635756, -2.7902472, 1.4153975, -3.8315115, 4.0538225
9: -1.5953801, 1.6335883, -1.7957036, 1.8860987, -3.4814787, 3.4292920

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.9874865, upper bound: 4.2416106
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.9874865, upper bound: 4.2416106
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.3413130, 1.2534575, -2.0767829, 1.7628329, -3.1041460, 3.3302405
1: -1.1064329, 1.0837612, -1.6826046, 1.5975116, -2.7039447, 2.7663658
2: -1.1643872, 1.2077436, -2.0168891, 1.6578529, -2.8222401, 3.2246327
3: -1.4114528, 0.9982265, -2.2930646, 1.4003248, -2.8117776, 3.2912910
4: -1.5771528, 1.1021525, -2.4442232, 1.6595405, -3.2366934, 3.5463758
5: -1.1708810, 1.2549249, -1.9167713, 1.7970213, -2.9679022, 3.1716962
6: -1.0917606, 1.3843800, -1.7280436, 1.9931232, -3.0848837, 3.1124234
7: -1.3048267, 1.3104151, -2.0280066, 2.0034316, -3.3082583, 3.3384218
8: -1.6211197, 1.2121111, -2.8265321, 1.4596353, -3.0807550, 4.0386434
9: -1.1512780, 1.3249016, -1.8049319, 1.9521161, -3.1033940, 3.1298335

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0969499, upper bound: 4.1826116
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.1454137, upper bound: 4.2357628
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.9482822, 1.6480703, -1.4782950, 1.3621676, -3.3104498, 3.1263652
1: -1.5734551, 1.5089953, -1.2167810, 1.1801137, -2.7535686, 2.7257762
2: -1.8592560, 1.5786078, -1.3164071, 1.2905837, -3.1498399, 2.8950148
3: -2.1378491, 1.3252633, -1.5852623, 1.0718790, -3.2097282, 2.9105256
4: -2.2920995, 1.5581020, -1.7476161, 1.2033216, -3.4954209, 3.3057179
5: -1.7776377, 1.7083184, -1.2994477, 1.3502386, -3.1278763, 3.0077660
6: -1.6119969, 1.8784479, -1.2085865, 1.4972289, -3.1092257, 3.0870342
7: -1.8976521, 1.8785890, -1.4390591, 1.4334061, -3.3310583, 3.3176482
8: -2.6199908, 1.3914015, -1.8619436, 1.2363417, -3.8563325, 3.2533450
9: -1.6874927, 1.8279969, -1.2595226, 1.4549198, -3.1424127, 3.0875194

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2369384, upper bound: 3.9343539
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2428917, upper bound: 4.0174864
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.9482822, 1.6480703, -2.0775809, 1.7635119, -3.7117941, 3.7256513
1: -1.5734551, 1.5089953, -1.6832691, 1.5980588, -3.1715140, 3.1922646
2: -1.8592560, 1.5786078, -2.0178633, 1.6583841, -3.5176401, 3.5964711
3: -2.1378491, 1.3252633, -2.2940154, 1.4007934, -3.5386424, 3.6192787
4: -2.2920995, 1.5581020, -2.4451375, 1.6603819, -3.9524813, 4.0032396
5: -1.7776377, 1.7083184, -1.9176168, 1.7976425, -3.5752802, 3.6259351
6: -1.6119969, 1.8784479, -1.7287744, 1.9938685, -3.6058655, 3.6072223
7: -1.8976521, 1.8785890, -2.0288177, 2.0042362, -3.9018884, 3.9074068
8: -2.6199908, 1.3914015, -2.8278096, 1.4600891, -4.0800800, 4.2192111
9: -1.6874927, 1.8279969, -1.8056450, 1.9527963, -3.6402891, 3.6336417

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2369384, upper bound: 4.2593121
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2428917, upper bound: 4.2634367
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1.9614283, 1.6498454, -2.0704732, 1.7321072, -3.6935353, 3.7203186
1: -1.5881400, 1.5179409, -1.6755357, 1.5946895, -3.1828294, 3.1934767
2: -1.8799704, 1.5837164, -2.0115166, 1.6540654, -3.5340357, 3.5952330
3: -2.1584244, 1.3336879, -2.2864597, 1.3964233, -3.5548477, 3.6201477
4: -2.3091495, 1.5712506, -2.4398639, 1.6581353, -3.9672847, 4.0111146
5: -1.7895476, 1.7037189, -1.9070085, 1.7859906, -3.5755382, 3.6107273
6: -1.6208949, 1.8778907, -1.7230678, 1.9743704, -3.5952654, 3.6009583
7: -1.9174019, 1.8962680, -2.0263715, 2.0018229, -3.9192247, 3.9226394
8: -2.6394985, 1.3785839, -2.8176849, 1.4459844, -4.0854826, 4.1962690
9: -1.6999354, 1.8343055, -1.8035028, 1.9407465, -3.6406820, 3.6378083

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2636720, upper bound: 4.2671583
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2805599, upper bound: 4.2804611
time: 3.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2141352, 1.8422754, -2.0704732, 1.7321072, -3.9462423, 3.9127486
1: -1.7850038, 1.6930523, -1.6755357, 1.5946895, -3.3796933, 3.3685880
2: -2.1852930, 1.7361182, -2.0115166, 1.6540654, -3.8393583, 3.7476349
3: -2.4509246, 1.4756417, -2.2864597, 1.3964233, -3.8473480, 3.7621014
4: -2.5976832, 1.7696438, -2.4398639, 1.6581353, -4.2558184, 4.2095079
5: -2.0583754, 1.8683832, -1.9070085, 1.7859906, -3.8443661, 3.7753916
6: -1.8486818, 2.0858793, -1.7230678, 1.9743704, -3.8230522, 3.8089471
7: -2.1710339, 2.1440139, -2.0263715, 2.0018229, -4.1728568, 4.1703854
8: -3.0206494, 1.4969856, -2.8176849, 1.4459844, -4.4666338, 4.3146706
9: -1.9298320, 2.0245910, -1.8035028, 1.9407465, -3.8705785, 3.8280938

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2679705, upper bound: 4.2654641
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2805599, upper bound: 4.2804611
time: 5.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.9614283, 1.6498454, -2.3249154, 1.9275001, -3.8889284, 3.9747608
1: -1.5881400, 1.5179409, -1.8744260, 1.7717481, -3.3598881, 3.3923669
2: -1.8799704, 1.5837164, -2.3204210, 1.8091429, -3.6891134, 3.9041374
3: -2.1584244, 1.3336879, -2.5819590, 1.5393851, -3.6978095, 3.9156470
4: -2.3091495, 1.5712506, -2.7309556, 1.8606935, -4.1698427, 4.3022060
5: -1.7895476, 1.7037189, -2.1773515, 1.9514437, -3.7409911, 3.8810704
6: -1.6208949, 1.8778907, -1.9532542, 2.1863575, -3.8072524, 3.8311448
7: -1.9174019, 1.8962680, -2.2835886, 2.2522337, -4.1696358, 4.1798568
8: -2.6394985, 1.3785839, -3.2014475, 1.5677909, -4.2072892, 4.5800314
9: -1.6999354, 1.8343055, -2.0347838, 2.1323524, -3.8322878, 3.8690894

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2635546, upper bound: 4.2671013
time: 2.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2667770, upper bound: 4.2617494
time: 2.06 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2667770, upper bound: 4.2805504
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2141352, 1.8422754, -2.3249154, 1.9275001, -4.1416354, 4.1671906
1: -1.7850038, 1.6930523, -1.8744260, 1.7717481, -3.5567517, 3.5674782
2: -2.1852930, 1.7361182, -2.3204210, 1.8091429, -3.9944358, 4.0565391
3: -2.4509246, 1.4756417, -2.5819590, 1.5393851, -3.9903097, 4.0576010
4: -2.5976832, 1.7696438, -2.7309556, 1.8606935, -4.4583769, 4.5005994
5: -2.0583754, 1.8683832, -2.1773515, 1.9514437, -4.0098190, 4.0457344
6: -1.8486818, 2.0858793, -1.9532542, 2.1863575, -4.0350394, 4.0391335
7: -2.1710339, 2.1440139, -2.2835886, 2.2522337, -4.4232674, 4.4276028
8: -3.0206494, 1.4969856, -3.2014475, 1.5677909, -4.5884404, 4.6984329
9: -1.9298320, 2.0245910, -2.0347838, 2.1323524, -4.0621843, 4.0593748

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2635546, upper bound: 4.2671013
time: 2.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803871, upper bound: 4.2803871
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -2.2722051, 1.8639841, -3.4487739, 3.6784186
1: -1.2903745, 1.2613496, -1.8348852, 1.7395056, -3.0298800, 3.0962348
2: -1.4349308, 1.3510394, -2.2685735, 1.7782106, -3.2131414, 3.6196129
3: -1.7076654, 1.1180336, -2.5290985, 1.5104836, -3.2181492, 3.6471322
4: -1.8715502, 1.2757896, -2.6726892, 1.8247923, -3.6963425, 3.9484787
5: -1.3935781, 1.4331366, -2.1233330, 1.9081000, -3.3016782, 3.5564694
6: -1.2973832, 1.5571899, -1.9074252, 2.1357093, -3.4330926, 3.4646151
7: -1.5392796, 1.5356119, -2.2403588, 2.2139995, -3.7532792, 3.7759707
8: -1.9922013, 1.1882241, -3.1243346, 1.5057561, -3.4979575, 4.3125587
9: -1.3524704, 1.4370781, -1.9910346, 2.0600424, -3.4125128, 3.4281127

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0472514, upper bound: 4.2440412
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0472514, upper bound: 4.2440412
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.8527009, 1.5966368, -2.2722051, 1.8639841, -3.7166851, 3.8688419
1: -1.4868313, 1.4444220, -1.8348852, 1.7395056, -3.2263370, 3.2793074
2: -1.7275662, 1.5125510, -2.2685735, 1.7782106, -3.5057769, 3.7811246
3: -2.0223882, 1.2615709, -2.5290985, 1.5104836, -3.5328717, 3.7906694
4: -2.1773813, 1.4731731, -2.6726892, 1.8247923, -4.0021734, 4.1458626
5: -1.6695801, 1.6089939, -2.1233330, 1.9081000, -3.5776801, 3.7323270
6: -1.5178220, 1.7667497, -1.9074252, 2.1357093, -3.6535313, 3.6741748
7: -1.7920233, 1.7922113, -2.2403588, 2.2139995, -4.0060229, 4.0325699
8: -2.4161141, 1.2641082, -3.1243346, 1.5057561, -3.9218702, 4.3884430
9: -1.5953801, 1.6336577, -1.9910346, 2.0600424, -3.6554224, 3.6246924

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9770145, upper bound: 4.2369101
time: 6.45 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9770145, upper bound: 4.2369101
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.5847899, 1.4062135, -2.5295198, 2.0545900, -3.6393800, 3.9357333
1: -1.2903745, 1.2613496, -2.0363009, 1.9171219, -3.2074964, 3.2976503
2: -1.4349308, 1.3510394, -2.5801153, 1.9356500, -3.3705807, 3.9311547
3: -1.7076654, 1.1180336, -2.8282173, 1.6534263, -3.3610916, 3.9462509
4: -1.8715502, 1.2757896, -2.9643507, 2.0291638, -3.9007139, 4.2401404
5: -1.3935781, 1.4331366, -2.3923757, 2.0796189, -3.4731970, 3.8255124
6: -1.2973832, 1.5571899, -2.1397996, 2.3474054, -3.6447887, 3.6969895
7: -1.5392796, 1.5356119, -2.5008836, 2.4654589, -4.0047383, 4.0364952
8: -1.9922013, 1.1882241, -3.5069590, 1.6304826, -3.6226840, 4.6951828
9: -1.3524704, 1.4370781, -2.2231030, 2.2567685, -3.6092389, 3.6601810

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.0274204, upper bound: 4.2411104
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.0274204, upper bound: 4.2661019
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8525033, 1.5966368, -2.5295198, 2.0545900, -3.9070933, 4.1261568
1: -1.4860272, 1.4444220, -2.0363009, 1.9171219, -3.4031491, 3.4807229
2: -1.7269343, 1.5125510, -2.5801153, 1.9356500, -3.6625843, 4.0926661
3: -2.0220840, 1.2615709, -2.8282173, 1.6534263, -3.6755104, 4.0897884
4: -2.1773813, 1.4720203, -2.9643507, 2.0291638, -4.2065449, 4.4363708
5: -1.6695801, 1.6086042, -2.3923757, 2.0796189, -3.7491989, 4.0009799
6: -1.5174130, 1.7667497, -2.1397996, 2.3474054, -3.8648186, 3.9065495
7: -1.7913712, 1.7922113, -2.5008836, 2.4654589, -4.2568302, 4.2930946
8: -2.4161141, 1.2635756, -3.5069590, 1.6304826, -4.0465965, 4.7705345
9: -1.5953801, 1.6335883, -2.2231030, 2.2567685, -3.8521485, 3.8566914

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -3.9728870, upper bound: 4.2365855
time: 1.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.9728870, upper bound: 4.2647055
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.9685943, 1.6801713, -2.5239367, 2.0639822, -4.0325766, 4.2041082
1: -1.5946360, 1.5217935, -2.0342703, 1.9155869, -3.5102229, 3.5560637
2: -1.8851464, 1.5873765, -2.5708663, 1.9446722, -3.8298187, 4.1582427
3: -2.1652603, 1.3378249, -2.8215320, 1.6531169, -3.8183773, 4.1593571
4: -2.3152921, 1.5730846, -2.9684014, 2.0280619, -4.3433542, 4.5414858
5: -1.7987158, 1.7147547, -2.3911114, 2.1052568, -3.9039726, 4.1058660
6: -1.6265180, 1.8967515, -2.1455412, 2.3684237, -3.9949417, 4.0422926
7: -1.9188622, 1.8981711, -2.4907839, 2.4548697, -4.3737321, 4.3889551
8: -2.6477284, 1.3918285, -3.5286887, 1.6835133, -4.3312416, 4.9205170
9: -1.7017889, 1.8460033, -2.2270379, 2.3100853, -4.0118742, 4.0730410

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 242

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

### Candidate
type: B, layer: 1, pos: 245

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 245

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.8940480, upper bound: 4.2586154
time: 2.02 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -3.9807577, upper bound: 4.2596647
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.9484388, 1.6404626, -2.5385122, 2.0662789, -4.0147176, 4.1789751
1: -1.5781724, 1.5089984, -2.0499606, 1.9254878, -3.5036602, 3.5589590
2: -1.8644561, 1.5756055, -2.5928035, 1.9515772, -3.8160334, 4.1684089
3: -2.1434865, 1.3263619, -2.8437948, 1.6622201, -3.8057065, 4.1701565
4: -2.2939630, 1.5611595, -2.9868734, 2.0424118, -4.3363748, 4.5480328
5: -1.7757763, 1.6946565, -2.4049220, 2.1038218, -3.8795981, 4.0995784
6: -1.6087468, 1.8667126, -2.1564331, 2.3710079, -3.9797547, 4.0231457
7: -1.9045069, 1.8835552, -2.5117409, 2.4735363, -4.3780432, 4.3952961
8: -2.6189747, 1.3714417, -3.5478897, 1.6748905, -4.2938652, 4.9193316
9: -1.6877439, 1.8233148, -2.2410514, 2.3180361, -4.0057802, 4.0643663

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2656831, upper bound: 4.2615933
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797644, upper bound: 4.2799034
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2011647, 1.8328401, -2.5385122, 2.0662789, -4.2674437, 4.3713522
1: -1.7748764, 1.6839421, -2.0499606, 1.9254878, -3.7003641, 3.7339027
2: -2.1697202, 1.7278454, -2.5928035, 1.9515772, -4.1212974, 4.3206491
3: -2.4357960, 1.4683250, -2.8437948, 1.6622201, -4.0980163, 4.3121200
4: -2.5824404, 1.7593064, -2.9868734, 2.0424118, -4.6248522, 4.7461796
5: -2.0445123, 1.8592123, -2.4049220, 2.1038218, -4.1483340, 4.2641344
6: -1.8365257, 2.0746493, -2.1564331, 2.3710079, -4.2075338, 4.2310824
7: -2.1580224, 2.1312456, -2.5117409, 2.4735363, -4.6315584, 4.6429863
8: -3.0001774, 1.4896569, -3.5478897, 1.6748905, -4.6750679, 5.0375466
9: -1.9176787, 2.0134921, -2.2410514, 2.3180361, -4.2357149, 4.2545433

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2656831, upper bound: 4.2615933
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797644, upper bound: 4.2799034
time: 2.76 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.9484388, 1.6404626, -2.7924316, 2.2534561, -4.2018948, 4.4328942
1: -1.5781724, 1.5089984, -2.2482793, 2.1008081, -3.6789806, 3.7572777
2: -1.8644561, 1.5756055, -2.9004233, 2.1074500, -3.9719062, 4.4760289
3: -2.1434865, 1.3263619, -3.1394947, 1.8035789, -3.9470654, 4.4658566
4: -2.2939630, 1.5611595, -3.2741139, 2.2465856, -4.5405483, 4.8352733
5: -1.7757763, 1.6946565, -2.6698141, 2.2723768, -4.0481529, 4.3644705
6: -1.6087468, 1.8667126, -2.3863821, 2.5809729, -4.1897197, 4.2530947
7: -1.9045069, 1.8835552, -2.7691746, 2.7219460, -4.6264529, 4.6527300
8: -2.6189747, 1.3714417, -3.9220977, 1.7971760, -4.4161506, 5.2935395
9: -1.6877439, 1.8233148, -2.4696856, 2.5105207, -4.1982646, 4.2930002

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2642819, upper bound: 4.2542330
time: 2.51 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794592, upper bound: 4.2797748
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2011647, 1.8328401, -2.7924316, 2.2534561, -4.4546208, 4.6252718
1: -1.7748764, 1.6839421, -2.2482793, 2.1008081, -3.8756845, 3.9322214
2: -2.1697202, 1.7278454, -2.9004233, 2.1074500, -4.2771702, 4.6282687
3: -2.4357960, 1.4683250, -3.1394947, 1.8035789, -4.2393751, 4.6078196
4: -2.5824404, 1.7593064, -3.2741139, 2.2465856, -4.8290262, 5.0334206
5: -2.0445123, 1.8592123, -2.6698141, 2.2723768, -4.3168888, 4.5290265
6: -1.8365257, 2.0746493, -2.3863821, 2.5809729, -4.4174986, 4.4610314
7: -2.1580224, 2.1312456, -2.7691746, 2.7219460, -4.8799686, 4.9004202
8: -3.0001774, 1.4896569, -3.9220977, 1.7971760, -4.7973533, 5.4117546
9: -1.9176787, 2.0134921, -2.4696856, 2.5105207, -4.4281993, 4.4831777

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 159

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2642819, upper bound: 4.2542330
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794592, upper bound: 4.2797748
time: 6.88 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.2722051, 1.8639841, -1.5847899, 1.4062135, -3.6784186, 3.4487739
1: -1.8348852, 1.7395056, -1.2903745, 1.2613496, -3.0962348, 3.0298800
2: -2.2685735, 1.7782106, -1.4349308, 1.3510394, -3.6196129, 3.2131414
3: -2.5290985, 1.5104836, -1.7076654, 1.1180336, -3.6471322, 3.2181492
4: -2.6726892, 1.8247923, -1.8715502, 1.2757896, -3.9484787, 3.6963425
5: -2.1233330, 1.9081000, -1.3935781, 1.4331366, -3.5564694, 3.3016782
6: -1.9074252, 2.1357093, -1.2973832, 1.5571899, -3.4646151, 3.4330926
7: -2.2403588, 2.2139995, -1.5392796, 1.5356119, -3.7759707, 3.7532792
8: -3.1243346, 1.5057561, -1.9922013, 1.1882241, -4.3125587, 3.4979575
9: -1.9910346, 2.0600424, -1.3524704, 1.4370781, -3.4281127, 3.4125128

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2440412, upper bound: 4.0472514
time: 2.09 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2440412, upper bound: 4.2619938
time: 2.45 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.2722051, 1.8639841, -1.8527009, 1.5966368, -3.8688419, 3.7166851
1: -1.8348852, 1.7395056, -1.4868313, 1.4444220, -3.2793074, 3.2263370
2: -2.2685735, 1.7782106, -1.7275662, 1.5125510, -3.7811246, 3.5057769
3: -2.5290985, 1.5104836, -2.0223882, 1.2615709, -3.7906694, 3.5328717
4: -2.6726892, 1.8247923, -2.1773813, 1.4731731, -4.1458626, 4.0021734
5: -2.1233330, 1.9081000, -1.6695801, 1.6089939, -3.7323270, 3.5776801
6: -1.9074252, 2.1357093, -1.5178220, 1.7667497, -3.6741748, 3.6535313
7: -2.2403588, 2.2139995, -1.7920233, 1.7922113, -4.0325699, 4.0060229
8: -3.1243346, 1.5057561, -2.4161141, 1.2641082, -4.3884430, 3.9218702
9: -1.9910346, 2.0600424, -1.5953801, 1.6336577, -3.6246924, 3.6554224

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2369101, upper bound: 3.9770145
time: 1.65 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2369101, upper bound: 4.2591637
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.5295198, 2.0545900, -1.5847899, 1.4062135, -3.9357333, 3.6393800
1: -2.0363009, 1.9171219, -1.2903745, 1.2613496, -3.2976503, 3.2074964
2: -2.5801153, 1.9356500, -1.4349308, 1.3510394, -3.9311547, 3.3705807
3: -2.8282173, 1.6534263, -1.7076654, 1.1180336, -3.9462509, 3.3610916
4: -2.9643507, 2.0291638, -1.8715502, 1.2757896, -4.2401404, 3.9007139
5: -2.3923757, 2.0796189, -1.3935781, 1.4331366, -3.8255124, 3.4731970
6: -2.1397996, 2.3474054, -1.2973832, 1.5571899, -3.6969895, 3.6447887
7: -2.5008836, 2.4654589, -1.5392796, 1.5356119, -4.0364952, 4.0047383
8: -3.5069590, 1.6304826, -1.9922013, 1.1882241, -4.6951828, 3.6226840
9: -2.2231030, 2.2567685, -1.3524704, 1.4370781, -3.6601810, 3.6092389

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -4.2411104, upper bound: 4.0274204
time: 1.70 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2411104, upper bound: 4.2603669
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.5295198, 2.0545900, -1.8525033, 1.5966368, -4.1261568, 3.9070933
1: -2.0363009, 1.9171219, -1.4860272, 1.4444220, -3.4807229, 3.4031491
2: -2.5801153, 1.9356500, -1.7269343, 1.5125510, -4.0926661, 3.6625843
3: -2.8282173, 1.6534263, -2.0220840, 1.2615709, -4.0897884, 3.6755104
4: -2.9643507, 2.0291638, -2.1773813, 1.4720203, -4.4363708, 4.2065449
5: -2.3923757, 2.0796189, -1.6695801, 1.6086042, -4.0009799, 3.7491989
6: -2.1397996, 2.3474054, -1.5174130, 1.7667497, -3.9065495, 3.8648186
7: -2.5008836, 2.4654589, -1.7913712, 1.7922113, -4.2930946, 4.2568302
8: -3.5069590, 1.6304826, -2.4161141, 1.2635756, -4.7705345, 4.0465965
9: -2.2231030, 2.2567685, -1.5953801, 1.6335883, -3.8566914, 3.8521485

Time for backsubstitution: 1.61 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.71 + 595.44 = 601.15 seconds
