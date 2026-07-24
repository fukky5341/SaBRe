## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 60.201135133499996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014)
1: (-17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989)
2: (-14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656)
3: (-16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100)
4: (-13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.70 + 2.34 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2372775, upper bound: 60.2372775

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2323519
time: 0.91 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.82 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2323519
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.4466705, 33.0043793, -11.9775000, 37.3109016, -47.7575607, 44.9818726
1: -14.8643341, 34.2352486, -16.9736977, 38.6834106, -53.5477371, 51.2089462
2: -12.7760410, 38.1106606, -14.5771618, 43.0234413, -55.7994843, 52.6878204
3: -13.9698524, 49.0998573, -15.9551287, 55.3119316, -69.2817688, 65.0549774
4: -12.0348778, 45.3377380, -13.6242332, 51.1773186, -63.2121964, 58.9619713

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
time: 1.10 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
time: 1.00 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.5646820, 36.3494148, -11.5280428, 36.0634918, -47.6281738, 47.8774529
1: -16.3725929, 37.6687965, -16.3616581, 37.3929443, -53.7655373, 54.0304565
2: -14.0860319, 41.9790382, -14.0542173, 41.5972443, -55.6832695, 56.0332565
3: -15.4223356, 53.9925766, -15.3825579, 53.4967842, -68.9191208, 69.3751373
4: -13.2208614, 49.8709221, -13.1590786, 49.4717827, -62.6926422, 63.0299988

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
time: 0.98 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
time: 0.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.56 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -10.4466705, 33.0043793, -10.4466705, 33.0043793, -43.4510307, 43.4510345
1: -14.8643341, 34.2352486, -14.8643341, 34.2352486, -49.0995827, 49.0995827
2: -12.7760410, 38.1106606, -12.7760410, 38.1106606, -50.8867035, 50.8866997
3: -13.9698524, 49.0998573, -13.9698524, 49.0998573, -63.0697060, 63.0697060
4: -12.0348778, 45.3377380, -12.0348778, 45.3377380, -57.3726120, 57.3726120

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2299276
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2323519
time: 1.11 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.4466705, 33.0043793, -11.5646820, 36.3494148, -46.7960739, 44.5690536
1: -14.8643341, 34.2352486, -16.3725929, 37.6687965, -52.5331306, 50.6078415
2: -12.7760410, 38.1106606, -14.0860319, 41.9790382, -54.7550774, 52.1966858
3: -13.9698524, 49.0998573, -15.4223356, 53.9925766, -67.9624252, 64.5221786
4: -12.0348778, 45.3377380, -13.2208614, 49.8709221, -61.9057999, 58.5585938

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2299276
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2323519
time: 0.77 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.4541883, 36.0198555, -9.6633167, 30.5207977, -41.9749832, 45.6831741
1: -16.2153759, 37.3288498, -13.7461500, 31.6990681, -47.9144440, 51.0749931
2: -13.9513102, 41.6049156, -11.8245182, 35.3118134, -49.2631226, 53.4294319
3: -15.2747631, 53.5085945, -12.9121599, 45.3516579, -60.6264191, 66.4207535
4: -13.0982933, 49.4271393, -11.1455908, 42.0009995, -55.0992928, 60.5727310

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
time: 0.84 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.1919031, 35.2112198, -17.7528858, 52.8149071, -64.0068130, 52.9641037
1: -15.8567715, 36.4924545, -24.7591095, 54.6754608, -70.5322342, 61.2515564
2: -13.6427794, 40.6649666, -21.1967220, 60.8025742, -74.4453506, 61.8616867
3: -14.9311733, 52.3047867, -23.3136482, 77.8044739, -92.7356491, 75.6184158
4: -12.8189030, 48.3158951, -19.4207764, 72.4624329, -85.2813339, 67.7366714

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2213024
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.27 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2299276
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2323519
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2299276
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2332639, upper bound: 60.2323519
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2213024
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 4, lower bound: -60.2304496, upper bound: 60.2304496

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -10.3254414, 32.6397896, -41.3389664, 38.0621185
1: -12.4092007, 28.8249302, -14.6937313, 33.8598900, -46.2690887, 43.5186615
2: -10.6860304, 32.1388817, -12.6306772, 37.6966248, -48.3826561, 44.7695580
3: -11.6377096, 41.3525887, -13.8074245, 48.5648079, -60.2025185, 55.1600113
4: -10.1306763, 38.2451363, -11.9017200, 44.8468323, -54.9775085, 50.1468506

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -10.2946529, 32.5462990, -49.1115570, 59.6405373
1: -23.0251637, 51.0942726, -14.6486740, 33.7615929, -56.7867546, 65.7429504
2: -19.7440033, 56.8110428, -12.5918560, 37.5855904, -57.3295937, 69.4029007
3: -21.7395668, 72.8528137, -13.7683611, 48.4216614, -70.1612167, 86.6211777
4: -18.1693535, 67.7313232, -11.8678837, 44.7114906, -62.8808441, 79.5992050

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2371454
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2371454
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -11.4541883, 36.0198555, -44.7190323, 39.1908684
1: -12.4092007, 28.8249302, -16.2153759, 37.3288498, -49.7380524, 45.0403061
2: -10.6860304, 32.1388817, -13.9513102, 41.6049156, -52.2909470, 46.0901909
3: -11.6377096, 41.3525887, -15.2747631, 53.5085945, -65.1463013, 56.6273499
4: -10.1306763, 38.2451363, -13.0982933, 49.4271393, -59.5578156, 51.3434296

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2299276
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2299276
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -11.1919031, 35.2112198, -51.7764854, 60.5377922
1: -23.0251637, 51.0942726, -15.8567715, 36.4924545, -59.5176125, 66.9510422
2: -19.7440033, 56.8110428, -13.6427794, 40.6649666, -60.4089699, 70.4538193
3: -21.7395668, 72.8528137, -14.9311733, 52.3047867, -74.0443497, 87.7839890
4: -18.1693535, 67.7313232, -12.8189030, 48.3158951, -66.4852448, 80.5502243

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2323519
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2323519
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -9.6633167, 30.5207977, -40.4719887, 41.2115097
1: -14.1082268, 32.7253380, -13.7461500, 31.6990681, -45.8072929, 46.4714813
2: -12.1443062, 36.5262375, -11.8245182, 35.3118134, -47.4561195, 48.3507538
3: -13.2896729, 46.9455948, -12.9121599, 45.3516579, -58.6413307, 59.8577538
4: -11.4511833, 43.4067154, -11.1455908, 42.0009995, -53.4521790, 54.5523071

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -9.6633167, 30.5207977, -46.6400986, 58.4024658
1: -22.4673023, 50.3988075, -13.7461500, 31.6990681, -54.1663704, 64.1449509
2: -19.2585125, 56.0775299, -11.8245182, 35.3118134, -54.5703239, 67.9020386
3: -21.2393341, 71.9838791, -12.9121599, 45.3516579, -66.5909882, 84.8960419
4: -17.8179779, 66.7789383, -11.1455908, 42.0009995, -59.8189774, 77.9245300

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -17.7528858, 52.8149071, -62.7660866, 49.3010788
1: -14.1082268, 32.7253380, -24.7591095, 54.6754608, -68.7836685, 57.4844398
2: -12.1443062, 36.5262375, -21.1967220, 60.8025742, -72.9468689, 57.7229614
3: -13.2896729, 46.9455948, -23.3136482, 77.8044739, -91.0941467, 70.2592316
4: -11.4511833, 43.4067154, -19.4207764, 72.4624329, -83.9136200, 62.8274918

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -17.7528858, 52.8149071, -68.9342041, 66.4920349
1: -22.4673023, 50.3988075, -24.7591095, 54.6754608, -77.1427536, 75.1579056
2: -19.2585125, 56.0775299, -21.1967220, 60.8025742, -80.0610886, 77.2742462
3: -21.2393341, 71.9838791, -23.3136482, 77.8044739, -99.0438080, 95.2975235
4: -17.8179779, 66.7789383, -19.4207764, 72.4624329, -90.2804108, 86.1997147

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.43 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2371454
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2371454
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2299276
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2299276
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2323519
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2323519
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2304496

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -8.6991749, 27.7366791, -36.4358521, 36.4358521
1: -12.4092007, 28.8249302, -12.4092007, 28.8249302, -41.2341309, 41.2341309
2: -10.6860304, 32.1388817, -10.6860304, 32.1388817, -42.8249130, 42.8249130
3: -11.6377096, 41.3525887, -11.6377096, 41.3525887, -52.9902954, 52.9902954
4: -10.1306763, 38.2451363, -10.1306763, 38.2451363, -48.3758125, 48.3758125

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350424
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2330853
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -16.4778557, 49.0316277, -57.7308044, 44.2145348
1: -12.4092007, 28.8249302, -22.8805523, 50.7702751, -63.1794662, 51.7054825
2: -10.6860304, 32.1388817, -19.6148682, 56.4666824, -67.1527100, 51.7537498
3: -11.6377096, 41.3525887, -21.6091194, 72.3687057, -84.0064163, 62.9617081
4: -10.1306763, 38.2451363, -18.0539436, 67.3143234, -77.4449997, 56.2990799

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350424
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2330853
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -8.6602230, 27.6217041, -44.1869659, 58.0061111
1: -23.0251637, 51.0942726, -12.3543396, 28.7050247, -51.7301865, 63.4486122
2: -19.7440033, 56.8110428, -10.6386271, 32.0058823, -51.7498856, 67.4496689
3: -21.7395668, 72.8528137, -11.5859261, 41.1813660, -62.9209328, 84.4387360
4: -18.1693535, 67.7313232, -10.0872440, 38.0864449, -56.2557983, 77.8185654

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -16.4778557, 49.0316277, -65.5968933, 65.8237457
1: -23.0251637, 51.0942726, -22.8805523, 50.7702751, -73.7954407, 73.9748230
2: -19.7440033, 56.8110428, -19.6148682, 56.4666824, -76.2106857, 76.4259109
3: -21.7395668, 72.8528137, -21.6091194, 72.3687057, -94.1082687, 94.4619217
4: -18.1693535, 67.7313232, -18.0539436, 67.3143234, -85.4836731, 85.7852631

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -9.9511909, 31.5481949, -40.2473679, 37.6878700
1: -12.4092007, 28.8249302, -14.1082268, 32.7253380, -45.1345367, 42.9331551
2: -10.6860304, 32.1388817, -12.1443062, 36.5262375, -47.2122688, 44.2831879
3: -11.6377096, 41.3525887, -13.2896729, 46.9455948, -58.5833054, 54.6422615
4: -10.1306763, 38.2451363, -11.4511833, 43.4067154, -53.5373917, 49.6963081

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2242382, upper bound: 60.2297568
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -16.1192989, 48.7391510, -57.4383240, 43.8559799
1: -12.4092007, 28.8249302, -22.4673023, 50.3988075, -62.8080025, 51.2922325
2: -10.6860304, 32.1388817, -19.2585125, 56.0775299, -66.7635498, 51.3973885
3: -11.6377096, 41.3525887, -21.2393341, 71.9838791, -83.6215897, 62.5919228
4: -10.1306763, 38.2451363, -17.8179779, 66.7789383, -76.9096069, 56.0631142

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2242382, upper bound: 60.2297568
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -9.9511909, 31.5481949, -48.1134567, 59.2970810
1: -23.0251637, 51.0942726, -14.1082268, 32.7253380, -55.7504959, 65.2024994
2: -19.7440033, 56.8110428, -12.1443062, 36.5262375, -56.2702408, 68.9553528
3: -21.7395668, 72.8528137, -13.2896729, 46.9455948, -68.6851654, 86.1424866
4: -18.1693535, 67.7313232, -11.4511833, 43.4067154, -61.5760689, 79.1825104

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2217609, upper bound: 60.2320112
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238214, upper bound: 60.2300780
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -16.1192989, 48.7391510, -65.3044128, 65.4651871
1: -23.0251637, 51.0942726, -22.4673023, 50.3988075, -73.4239731, 73.5615768
2: -19.7440033, 56.8110428, -19.2585125, 56.0775299, -75.8215332, 76.0695572
3: -21.7395668, 72.8528137, -21.2393341, 71.9838791, -93.7234421, 94.0921478
4: -18.1693535, 67.7313232, -17.8179779, 66.7789383, -84.9482880, 85.5493011

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2217609, upper bound: 60.2320112
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2217609, upper bound: 60.2300780
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -8.6991749, 27.7366791, -37.6878700, 40.2473679
1: -14.1082268, 32.7253380, -12.4092007, 28.8249302, -42.9331551, 45.1345367
2: -12.1443062, 36.5262375, -10.6860304, 32.1388817, -44.2831879, 47.2122688
3: -13.2896729, 46.9455948, -11.6377096, 41.3525887, -54.6422615, 58.5833054
4: -11.4511833, 43.4067154, -10.1306763, 38.2451363, -49.6963081, 53.5373917

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1913764, upper bound: 60.1808957
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -9.9511909, 31.5481949, -41.4993858, 41.4993858
1: -14.1082268, 32.7253380, -14.1082268, 32.7253380, -46.8335609, 46.8335571
2: -12.1443062, 36.5262375, -12.1443062, 36.5262375, -48.6705399, 48.6705399
3: -13.2896729, 46.9455948, -13.2896729, 46.9455948, -60.2352676, 60.2352676
4: -11.4511833, 43.4067154, -11.4511833, 43.4067154, -54.8578911, 54.8578911

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1913764, upper bound: 60.1808957
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -8.6991749, 27.7366791, -43.8559799, 57.4383240
1: -22.4673023, 50.3988075, -12.4092007, 28.8249302, -51.2922325, 62.8080025
2: -19.2585125, 56.0775299, -10.6860304, 32.1388817, -51.3973885, 66.7635498
3: -21.2393341, 71.9838791, -11.6377096, 41.3525887, -62.5919228, 83.6215897
4: -17.8179779, 66.7789383, -10.1306763, 38.2451363, -56.0631142, 76.9096069

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2181572, upper bound: 60.2254252
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2192095, upper bound: 60.2283567
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -9.9511909, 31.5481949, -47.6674919, 58.6903419
1: -22.4673023, 50.3988075, -14.1082268, 32.7253380, -55.1926422, 64.5070190
2: -19.2585125, 56.0775299, -12.1443062, 36.5262375, -55.7847481, 68.2218170
3: -21.2393341, 71.9838791, -13.2896729, 46.9455948, -68.1849289, 85.2735443
4: -17.8179779, 66.7789383, -11.4511833, 43.4067154, -61.2246933, 78.2301178

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2181572, upper bound: 60.2254252
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2192095, upper bound: 60.2283567
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -16.7040405, 49.7543106, -59.7055016, 48.2522354
1: -14.1082268, 32.7253380, -23.2350788, 51.5204048, -65.6286240, 55.9604073
2: -12.1443062, 36.5262375, -19.9190445, 57.2917175, -69.4360046, 56.4452782
3: -13.2896729, 46.9455948, -21.9273682, 73.4507294, -86.7404022, 68.8729630
4: -11.4511833, 43.4067154, -18.3297539, 68.3172836, -79.7684631, 61.7364693

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274722, upper bound: 60.2179560
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2300869, upper bound: 60.2209397
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -16.3657169, 49.3871498, -59.3383408, 47.9139099
1: -14.1082268, 32.7253380, -22.8206711, 51.0731087, -65.1813278, 55.5460014
2: -12.1443062, 36.5262375, -19.5613575, 56.8234863, -68.9677811, 56.0875931
3: -13.2896729, 46.9455948, -21.5540047, 72.9262695, -86.2159271, 68.4996033
4: -11.4511833, 43.4067154, -18.0741940, 67.6964035, -79.1475830, 61.4809074

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274722, upper bound: 60.2179560
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2300869, upper bound: 60.2209397
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -16.7040405, 49.7543106, -65.8736115, 65.4431915
1: -22.4673023, 50.3988075, -23.2350788, 51.5204048, -73.9877090, 73.6338806
2: -19.2585125, 56.0775299, -19.9190445, 57.2917175, -76.5502243, 75.9965744
3: -21.2393341, 71.9838791, -21.9273682, 73.4507294, -94.6900635, 93.9112473
4: -17.8179779, 66.7789383, -18.3297539, 68.3172836, -86.1352615, 85.1086884

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2300985
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212967, upper bound: 60.2272077
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -16.3657169, 49.3871498, -65.5064468, 65.1048660
1: -22.4673023, 50.3988075, -22.8206711, 51.0731087, -73.5404129, 73.2194672
2: -19.2585125, 56.0775299, -19.5613575, 56.8234863, -76.0819931, 75.6388855
3: -21.2393341, 71.9838791, -21.5540047, 72.9262695, -94.1655884, 93.5378876
4: -17.8179779, 66.7789383, -18.0741940, 67.6964035, -85.5143814, 84.8531189

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2300985
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212967, upper bound: 60.2272077
time: 1.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.14 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350424
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2330853
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350424
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2330853
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2258145, upper bound: 60.2186772
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2242382, upper bound: 60.2297568
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2242382, upper bound: 60.2297568
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2217609, upper bound: 60.2320112
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2238214, upper bound: 60.2300780
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2217609, upper bound: 60.2320112
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2217609, upper bound: 60.2300780
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.1913764, upper bound: 60.1808957
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.1913764, upper bound: 60.1808957
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2181572, upper bound: 60.2254252
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2192095, upper bound: 60.2283567
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2181572, upper bound: 60.2254252
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2192095, upper bound: 60.2283567
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2274722, upper bound: 60.2179560
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2300869, upper bound: 60.2209397
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2274722, upper bound: 60.2179560
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2300869, upper bound: 60.2209397
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2300985
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2212967, upper bound: 60.2272077
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2300985
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 4, lower bound: -60.2212967, upper bound: 60.2272077

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.4733706, 24.1488380, -8.5147905, 27.2209606, -34.6943321, 32.6636276
1: -10.6540947, 25.1270370, -12.1482782, 28.2915478, -38.9456367, 37.2753143
2: -9.2093201, 28.1137600, -10.4640541, 31.5510159, -40.7603378, 38.5778122
3: -9.9912634, 36.2262306, -11.3983784, 40.6140099, -50.6052742, 47.6246071
4: -8.8301640, 33.4773788, -9.9365444, 37.5493355, -46.3794937, 43.4139252

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2246799, upper bound: 60.2331079
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2246799, upper bound: 60.2354232
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.4589252, 27.0919533, -8.6991749, 27.7366791, -36.1956024, 35.7911263
1: -12.0820923, 28.1615906, -12.4092007, 28.8249302, -40.9070206, 40.5707932
2: -10.4120913, 31.3883038, -10.6860304, 32.1388817, -42.5509720, 42.0743332
3: -11.3392544, 40.4026985, -11.6377096, 41.3525887, -52.6918373, 52.0404091
4: -9.8867702, 37.3420067, -10.1306763, 38.2451363, -48.1319008, 47.4726830

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2319721
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2334250
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.4733706, 24.1488380, -16.2410069, 48.3785133, -55.8518791, 40.3898468
1: -10.6540947, 25.1270370, -22.5299034, 50.0907173, -60.7448120, 47.6569405
2: -9.2093201, 28.1137600, -19.3242035, 55.7209778, -64.9302979, 47.4379578
3: -9.9912634, 36.2262306, -21.3023643, 71.4364471, -81.4277115, 57.5285950
4: -8.8301640, 33.4773788, -17.8094425, 66.4158707, -75.2460327, 51.2868195

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2252744
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2317797
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.4589252, 27.0919533, -16.4778557, 49.0316277, -57.4905548, 43.5698090
1: -12.0820923, 28.1615906, -22.8805523, 50.7702751, -62.8523674, 51.0421448
2: -10.4120913, 31.3883038, -19.6148682, 56.4666824, -66.8787766, 51.0031738
3: -11.3392544, 40.4026985, -21.6091194, 72.3687057, -83.7079620, 62.0118179
4: -9.8867702, 37.3420067, -18.0539436, 67.3143234, -77.2010956, 55.3959503

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2234286
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2281544
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.1109037, 41.8675880, -8.4260359, 26.9165096, -41.0274124, 50.2936249
1: -19.5679359, 43.4133492, -12.0202808, 27.9797115, -47.5476456, 55.4336319
2: -16.7519665, 48.3994026, -10.3521919, 31.2062302, -47.9581909, 58.7515869
3: -18.5002823, 61.5973701, -11.2752352, 40.1403427, -58.6406250, 72.8725891
4: -15.4016666, 57.6641388, -9.8280153, 37.1332588, -52.5349274, 67.4921570

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2252744, upper bound: 60.2185265
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2234286, upper bound: 60.2185023
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.9450722, 53.3562126, -8.2855310, 26.6233082, -44.5683746, 61.6417427
1: -24.9718819, 55.2171478, -11.8308735, 27.6709976, -52.6428795, 67.0480194
2: -21.3408375, 61.3745003, -10.1812363, 30.8620415, -52.2028732, 71.5557404
3: -23.5634995, 78.4572372, -11.1230497, 39.7231178, -63.2866173, 89.5802841
4: -19.5413036, 73.1064529, -9.6848831, 36.7162781, -56.2575722, 82.7913284

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2252744, upper bound: 60.2281338
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2234286, upper bound: 60.2281097
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.1109037, 41.8675880, -16.2849541, 48.4252548, -62.5361595, 58.1525345
1: -19.5679359, 43.4133492, -22.6085014, 50.1475334, -69.7154694, 66.0218506
2: -16.7519665, 48.3994026, -19.3789940, 55.7847061, -72.5366440, 67.7783813
3: -18.5002823, 61.5973701, -21.3541279, 71.4672775, -89.9675598, 82.9514999
4: -15.4016666, 57.6641388, -17.8308887, 66.5087128, -81.9103775, 75.4950256

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2143773
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2186772
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.9450722, 53.3562126, -16.0569363, 47.9093857, -65.8544617, 69.4131393
1: -24.9718819, 55.2171478, -22.2776299, 49.6034279, -74.5753098, 77.4947815
2: -21.3408375, 61.3745003, -19.0907478, 55.1756859, -76.5165253, 80.4652481
3: -23.5634995, 78.4572372, -21.0765228, 70.7361603, -94.2996597, 99.5337601
4: -19.5413036, 73.1064529, -17.6152496, 65.7634125, -85.3047180, 90.7216949

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2303308
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2355284
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.9862747, 25.6116962, -9.9511909, 31.5481949, -39.5344696, 35.5628891
1: -11.4230928, 26.6387024, -14.1082268, 32.7253380, -44.1484299, 40.7469292
2: -9.8432503, 29.7230167, -12.1443062, 36.5262375, -46.3694878, 41.8673172
3: -10.7079391, 38.2268753, -13.2896729, 46.9455948, -57.6535339, 51.5165482
4: -9.3737078, 35.3655739, -11.4511833, 43.4067154, -52.7804184, 46.8167496

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198862, upper bound: 60.2273044
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198862, upper bound: 60.2292907
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.4350300, 29.5662556, -9.9022808, 31.4042149, -40.8392448, 39.4685364
1: -13.3337088, 30.6914234, -14.0401669, 32.5758514, -45.9095612, 44.7315903
2: -11.4997158, 34.2629051, -12.0863848, 36.3619347, -47.8616486, 46.3492889
3: -12.5133295, 44.0571518, -13.2255030, 46.7348862, -59.2482147, 57.2826538
4: -10.8592606, 40.8404045, -11.3998661, 43.2122345, -54.0714951, 52.2402725

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198862, upper bound: 60.2247163
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2228698, upper bound: 60.2257674
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.9862747, 25.6116962, -16.1192989, 48.7391510, -56.7254257, 41.7309952
1: -11.4230928, 26.6387024, -22.4673023, 50.3988075, -61.8218956, 49.1060028
2: -9.8432503, 29.7230167, -19.2585125, 56.0775299, -65.9207764, 48.9815216
3: -10.7079391, 38.2268753, -21.2393341, 71.9838791, -82.6918182, 59.4662056
4: -9.3737078, 35.3655739, -17.8179779, 66.7789383, -76.1526413, 53.1835518

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2157055, upper bound: 60.2203534
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2293597, upper bound: 60.2217632
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.4350300, 29.5662556, -16.0599804, 48.5664787, -58.0015030, 45.6262360
1: -13.3337088, 30.6914234, -22.3833942, 50.2205276, -63.5542374, 53.0748177
2: -11.4997158, 34.2629051, -19.1867943, 55.8817711, -67.3814850, 53.4496994
3: -12.5133295, 44.0571518, -21.1627502, 71.7335663, -84.2468872, 65.2198944
4: -10.8592606, 40.8404045, -17.7557144, 66.5460663, -77.4053268, 58.5961189

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158211, upper bound: 60.2144646
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2273896, upper bound: 60.2137625
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.8627434, 44.4193230, -9.7795925, 31.0500259, -45.9127579, 54.1989136
1: -20.4770241, 46.0121498, -13.8654299, 32.2109718, -52.6879959, 59.8775749
2: -17.6456852, 51.2711029, -11.9404449, 35.9571266, -53.6028137, 63.2115402
3: -19.5267849, 65.8024368, -13.0610533, 46.2243652, -65.7511444, 78.8634872
4: -16.3723183, 61.1303329, -11.2711277, 42.7309723, -59.1032791, 72.4014587

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158201, upper bound: 60.2293524
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201887, upper bound: 60.2315827
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16.3341827, 48.6779442, -9.9511909, 31.5481949, -47.8823776, 58.6291351
1: -22.6960735, 50.4090309, -14.1082268, 32.7253380, -55.4214058, 64.5172501
2: -19.4665813, 56.0498695, -12.1443062, 36.5262375, -55.9928207, 68.1941528
3: -21.4443798, 71.8867035, -13.2896729, 46.9455948, -68.3899689, 85.1763687
4: -17.9188328, 66.8347778, -11.4511833, 43.4067154, -61.3255463, 78.2859650

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158201, upper bound: 60.2282205
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226647, upper bound: 60.2296409
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.8627434, 44.4193230, -15.9007559, 48.1240730, -62.9868126, 60.3200798
1: -20.4770241, 46.0121498, -22.1525993, 49.7623520, -70.2393646, 68.1647339
2: -17.6456852, 51.2711029, -18.9927578, 55.3783035, -73.0239868, 70.2638474
3: -19.5267849, 65.8024368, -20.9553261, 71.1024780, -90.6292419, 86.7577667
4: -16.3723183, 61.1303329, -17.5896587, 65.9407654, -82.3130798, 78.7199936

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2163169
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2300780
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.3341827, 48.6779442, -16.1192989, 48.7391510, -65.0733337, 64.7972412
1: -22.6960735, 50.4090309, -22.4673023, 50.3988075, -73.0948715, 72.8763351
2: -19.4665813, 56.0498695, -19.2585125, 56.0775299, -75.5440979, 75.3083801
3: -21.4443798, 71.8867035, -21.2393341, 71.9838791, -93.4282455, 93.1260300
4: -17.9188328, 66.8347778, -17.8179779, 66.7789383, -84.6977539, 84.6527557

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2163169
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2188254, upper bound: 60.2250373
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198138, upper bound: 60.2261004
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.7370825, 30.8880539, -8.6991749, 27.7366791, -37.4737625, 39.5872231
1: -13.8120041, 32.0481262, -12.4092007, 28.8249302, -42.6369324, 44.4573288
2: -11.8939190, 35.7687950, -10.6860304, 32.1388817, -44.0327988, 46.4548264
3: -13.0009775, 45.9720306, -11.6377096, 41.3525887, -54.3535614, 57.6097412
4: -11.2234163, 42.5065231, -10.1306763, 38.2451363, -49.4685516, 52.6371994

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2164519
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2240234
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.7370825, 30.8880539, -9.9511909, 31.5481949, -41.2852783, 40.8392372
1: -13.8120041, 32.0481262, -14.1082268, 32.7253380, -46.5373421, 46.1563530
2: -11.8939190, 35.7687950, -12.1443062, 36.5262375, -48.4201546, 47.9130936
3: -13.0009775, 45.9720306, -13.2896729, 46.9455948, -59.9465714, 59.2617035
4: -11.2234163, 42.5065231, -11.4511833, 43.4067154, -54.6301308, 53.9576988

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1808957, upper bound: 60.1913764
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1808957, upper bound: 60.2210875
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.6605654, 47.3928490, -8.6991749, 27.7366791, -43.3972435, 56.0920258
1: -21.8332367, 49.0203247, -12.4092007, 28.8249302, -50.6581650, 61.4295235
2: -18.7182560, 54.5562439, -10.6860304, 32.1388817, -50.8571396, 65.2422714
3: -20.6293983, 70.0185394, -11.6377096, 41.3525887, -61.9819870, 81.6562500
4: -17.3239307, 64.9725037, -10.1306763, 38.2451363, -55.5690651, 75.1031799

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196256, upper bound: 60.2263092
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196256, upper bound: 60.2263092
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.0755749, 45.9713211, -8.6991749, 27.7366791, -42.8122559, 54.6704941
1: -21.0839806, 47.5322456, -12.4092007, 28.8249302, -49.9089127, 59.9414444
2: -18.0686455, 52.9041939, -10.6860304, 32.1388817, -50.2075272, 63.5902252
3: -19.9506207, 68.0000534, -11.6377096, 41.3525887, -61.3032074, 79.6377640
4: -16.7616405, 63.0414658, -10.1306763, 38.2451363, -55.0067749, 73.1721420

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2292441
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2292441
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.6605654, 47.3928490, -9.9511909, 31.5481949, -47.2087593, 57.3440399
1: -21.8332367, 49.0203247, -14.1082268, 32.7253380, -54.5585747, 63.1285515
2: -18.7182560, 54.5562439, -12.1443062, 36.5262375, -55.2444916, 66.7005310
3: -20.6293983, 70.0185394, -13.2896729, 46.9455948, -67.5749893, 83.3082123
4: -17.3239307, 64.9725037, -11.4511833, 43.4067154, -60.7306442, 76.4236832

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172664, upper bound: 60.2243724
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172664, upper bound: 60.2254252
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.0755749, 45.9713211, -9.9511909, 31.5481949, -46.6237717, 55.9225121
1: -21.0839806, 47.5322456, -14.1082268, 32.7253380, -53.8093185, 61.6404572
2: -18.0686455, 52.9041939, -12.1443062, 36.5262375, -54.5948830, 65.0484924
3: -19.9506207, 68.0000534, -13.2896729, 46.9455948, -66.8962173, 81.2897263
4: -16.7616405, 63.0414658, -11.4511833, 43.4067154, -60.1683540, 74.4926453

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183186, upper bound: 60.2273034
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183186, upper bound: 60.2283567
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -16.7040405, 49.7543106, -58.8516350, 45.7680588
1: -12.8702602, 30.1397648, -23.2350788, 51.5204048, -64.3906631, 53.3748360
2: -11.0743856, 33.6567726, -19.9190445, 57.2917175, -68.3661041, 53.5758133
3: -12.1959352, 43.2404213, -21.9273682, 73.4507294, -85.6466599, 65.1677856
4: -10.4772635, 40.0022278, -18.3297539, 68.3172836, -78.7945328, 58.3319817

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2092221, upper bound: 60.1986673
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2259560, upper bound: 60.2169543
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2278274, upper bound: 60.2177256
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -16.6469154, 49.6177139, -60.4873428, 51.0553055
1: -15.3343544, 35.6384964, -23.1504688, 51.3778954, -66.7122345, 58.7889633
2: -13.1785946, 39.7972412, -19.8421555, 57.1343842, -70.3129730, 59.6393967
3: -14.4879837, 51.1623955, -21.8627911, 73.2561569, -87.7441406, 73.0251846
4: -12.3919563, 47.3149300, -18.2700100, 68.1302109, -80.5221405, 65.5849380

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286810, upper bound: 60.2199362
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2260622, upper bound: 60.2207075
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -16.3657169, 49.3871498, -58.4844742, 45.4297447
1: -12.8702602, 30.1397648, -22.8206711, 51.0731087, -63.9433556, 52.9604340
2: -11.0743856, 33.6567726, -19.5613575, 56.8234863, -67.8978729, 53.2181320
3: -12.1959352, 43.2404213, -21.5540047, 72.9262695, -85.1222000, 64.7944260
4: -10.4772635, 40.0022278, -18.0741940, 67.6964035, -78.1736526, 58.0764236

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2230103, upper bound: 60.2148587
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2254126, upper bound: 60.2158284
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -16.3081093, 49.2572975, -60.1269302, 50.7165031
1: -15.3343544, 35.6384964, -22.7371349, 50.9363861, -66.2707367, 58.3756332
2: -13.1785946, 39.7972412, -19.4859238, 56.6739273, -69.8525162, 59.2831612
3: -14.4879837, 51.1623955, -21.4895992, 72.7386627, -87.2266464, 72.6519928
4: -12.3919563, 47.3149300, -18.0153370, 67.5142441, -79.9061890, 65.3302612

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250279, upper bound: 60.2178423
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2279593, upper bound: 60.2188121
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.6242218, 44.3043480, -16.4577980, 49.0659981, -63.6902199, 60.7621460
1: -20.2457829, 45.8350677, -22.8685036, 50.8067131, -71.0524979, 68.7035675
2: -17.4124851, 51.0891838, -19.6141090, 56.5056267, -73.9181137, 70.7032928
3: -19.2638702, 65.6291351, -21.6074181, 72.4687347, -91.7326050, 87.2365570
4: -16.2089767, 60.8570938, -18.0722771, 67.3752670, -83.5842438, 78.9293671

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2059931, upper bound: 60.2244252
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156005, upper bound: 60.2292721
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.8565083, 47.9964638, -16.7040405, 49.7543106, -65.6108093, 64.7004929
1: -22.0979061, 49.6354141, -23.2350788, 51.5204048, -73.6183090, 72.8704910
2: -18.9476433, 55.2243576, -19.9190445, 57.2917175, -76.2393570, 75.1434021
3: -20.9038620, 70.9059448, -21.9273682, 73.4507294, -94.3545837, 92.8332977
4: -17.5394630, 65.7744293, -18.3297539, 68.3172836, -85.8567429, 84.1041870

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104263, upper bound: 60.2194369
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2200025, upper bound: 60.2241022
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.6242218, 44.3043480, -16.1362915, 48.7514076, -63.3756294, 60.4406395
1: -20.2457829, 45.8350677, -22.4888077, 50.4138947, -70.6596756, 68.3238754
2: -17.4124851, 51.0891838, -19.2826118, 56.0998726, -73.5123596, 70.3717957
3: -19.2638702, 65.6291351, -21.2581043, 72.0144501, -91.2783203, 86.8872223
4: -16.2089767, 60.8570938, -17.8358879, 66.8282547, -83.0372238, 78.6929779

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2049972, upper bound: 60.2226399
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2116103, upper bound: 60.2228429
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.8565083, 47.9964638, -16.3657169, 49.3871498, -65.2436523, 64.3621826
1: -22.0979061, 49.6354141, -22.8206711, 51.0731087, -73.1710129, 72.4560852
2: -18.9476433, 55.2243576, -19.5613575, 56.8234863, -75.7711258, 74.7857132
3: -20.9038620, 70.9059448, -21.5540047, 72.9262695, -93.8301163, 92.4599380
4: -17.5394630, 65.7744293, -18.0741940, 67.6964035, -85.2358704, 83.8486176

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163905, upper bound: 60.2163169
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163905, upper bound: 60.2272077
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.73 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2246799, upper bound: 60.2331079
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2246799, upper bound: 60.2354232
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2319721
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2334250
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2252744
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2317797
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2234286
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2281544
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2252744, upper bound: 60.2185265
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2234286, upper bound: 60.2185023
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2252744, upper bound: 60.2281338
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2234286, upper bound: 60.2281097
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2143773
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2186772
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2303308
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2355284
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2198862, upper bound: 60.2273044
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2198862, upper bound: 60.2292907
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2198862, upper bound: 60.2247163
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2228698, upper bound: 60.2257674
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2157055, upper bound: 60.2203534
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2293597, upper bound: 60.2217632
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2158211, upper bound: 60.2144646
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2273896, upper bound: 60.2137625
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2158201, upper bound: 60.2293524
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2201887, upper bound: 60.2315827
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2158201, upper bound: 60.2282205
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2226647, upper bound: 60.2296409
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2163169
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2163169, upper bound: 60.2300780
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2188254, upper bound: 60.2250373
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2198138, upper bound: 60.2261004
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2164519
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2240234
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.1808957, upper bound: 60.1913764
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.1808957, upper bound: 60.2210875
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2196256, upper bound: 60.2263092
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2196256, upper bound: 60.2263092
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2292441
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2292441
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2172664, upper bound: 60.2243724
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2172664, upper bound: 60.2254252
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2183186, upper bound: 60.2273034
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2183186, upper bound: 60.2283567
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2259560, upper bound: 60.2169543
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2278274, upper bound: 60.2177256
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2286810, upper bound: 60.2199362
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2260622, upper bound: 60.2207075
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2230103, upper bound: 60.2148587
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2254126, upper bound: 60.2158284
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2250279, upper bound: 60.2178423
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2279593, upper bound: 60.2188121
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2059931, upper bound: 60.2244252
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2156005, upper bound: 60.2292721
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2104263, upper bound: 60.2194369
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2200025, upper bound: 60.2241022
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2049972, upper bound: 60.2226399
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2116103, upper bound: 60.2228429
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2163905, upper bound: 60.2163169
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 4, lower bound: -60.2163905, upper bound: 60.2272077

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.4733706, 24.1488380, -7.6937361, 23.9664497, -31.4398193, 31.8425732
1: -10.6540947, 25.1270370, -10.9899483, 25.0429764, -35.6970711, 36.1169853
2: -9.2093201, 28.1137600, -9.5338907, 27.9177094, -37.1270256, 37.6476517
3: -9.9912634, 36.2262306, -10.1698980, 35.7666893, -45.7579536, 46.3961296
4: -8.8301640, 33.4773788, -9.0573406, 33.2245636, -42.0547180, 42.5347214

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2232579, upper bound: 60.2325023
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.4733706, 24.1488380, -8.3547344, 26.7673855, -34.2407494, 32.5035706
1: -10.6540947, 25.1270370, -11.9245319, 27.8209152, -38.4750099, 37.0515671
2: -9.2093201, 28.1137600, -10.2704678, 31.0293274, -40.2386475, 38.3842278
3: -9.9912634, 36.2262306, -11.1917734, 39.9468613, -49.9381256, 47.4180031
4: -8.8301640, 33.4773788, -9.7622013, 36.9261665, -45.7563248, 43.2395782

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2219682, upper bound: 60.2291033
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2280992, upper bound: 60.2336509
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291491, upper bound: 60.2338100
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2292948, upper bound: 60.2335783
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.4589252, 27.0919533, -7.4733706, 24.1488380, -32.6077614, 34.5653229
1: -12.0820923, 28.1615906, -10.6540947, 25.1270370, -37.2091293, 38.8156853
2: -10.4120913, 31.3883038, -9.2093201, 28.1137600, -38.5258522, 40.5976257
3: -11.3392544, 40.4026985, -9.9912634, 36.2262306, -47.5654793, 50.3939629
4: -9.8867702, 37.3420067, -8.8301640, 33.4773788, -43.3641472, 46.1721687

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2326945, upper bound: 60.2246799
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334013, upper bound: 60.2319485
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.4589252, 27.0919533, -8.4589252, 27.0919533, -35.5508766, 35.5508766
1: -12.0820923, 28.1615906, -12.0820923, 28.1615906, -40.2436829, 40.2436829
2: -10.4120913, 31.3883038, -10.4120913, 31.3883038, -41.8003960, 41.8003960
3: -11.3392544, 40.4026985, -11.3392544, 40.4026985, -51.7419472, 51.7419472
4: -9.8867702, 37.3420067, -9.8867702, 37.3420067, -47.2287750, 47.2287750

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2326945, upper bound: 60.2261504
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334013, upper bound: 60.2319485
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.2477555, 23.4809265, -13.7637329, 40.7737389, -48.0214958, 37.2446594
1: -10.3286304, 24.4362679, -19.0361233, 42.2805214, -52.6091537, 43.4723892
2: -8.9293432, 27.3520660, -16.2971249, 47.1798096, -56.1091499, 43.6491890
3: -9.6945229, 35.2405243, -18.0353546, 60.0240097, -69.7185287, 53.2758713
4: -8.5797596, 32.5675926, -15.0064383, 56.2157898, -64.7955475, 47.5740318

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2249880
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154153, upper bound: 60.2197981
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166134, upper bound: 60.2247762
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163709, upper bound: 60.2235861
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154364, upper bound: 60.2221333
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152922, upper bound: 60.2222616
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.2254648, 23.4716644, -17.5134106, 52.0222282, -59.2476883, 40.9850769
1: -10.3174648, 24.4269600, -24.2994671, 53.8350220, -64.1524811, 48.7264252
2: -8.9150581, 27.3323803, -20.7623653, 59.8632202, -68.7782745, 48.0947380
3: -9.6869059, 35.2176857, -22.9770489, 76.5224457, -86.2093506, 58.1947327
4: -8.5678310, 32.5385628, -19.0530453, 71.3281708, -79.8960037, 51.5916061

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2296050
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226735, upper bound: 60.2222402
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166134, upper bound: 60.2292044
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163709, upper bound: 60.2289832
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250757, upper bound: 60.2288231
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2248943, upper bound: 60.2280087
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.2349100, 26.4188538, -14.0264816, 41.5607224, -49.7956314, 40.4453354
1: -11.7623854, 27.4692841, -19.4275627, 43.0964813, -54.8588676, 46.8968468
2: -10.1377096, 30.6259346, -16.6255379, 48.0640945, -58.2017975, 47.2514725
3: -11.0431681, 39.4105530, -18.3753910, 61.1247749, -72.1679459, 57.7859421
4: -9.6385145, 36.4333534, -15.2870178, 57.2594147, -66.8979263, 51.7203712

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2232131
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162423, upper bound: 60.2166351
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.0751925, 26.0635090, -17.7583504, 52.7161827, -60.7913704, 43.8218460
1: -11.5458527, 27.0962582, -24.6639824, 54.5493889, -66.0952454, 51.7602386
2: -9.9440928, 30.2098122, -21.0661926, 60.6673622, -70.6114426, 51.2760048
3: -10.8628225, 38.9008713, -23.2926273, 77.4969101, -88.3597183, 62.1934967
4: -9.4746342, 35.9310608, -19.3064880, 72.2635345, -81.7381668, 55.2375488

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2261480
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162423, upper bound: 60.2194586
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.8441534, 41.0654488, -7.1809878, 23.2799606, -37.1241150, 48.2464371
1: -19.1699409, 42.5819435, -10.2341194, 24.2265911, -43.3965302, 52.8160629
2: -16.4177227, 47.4986115, -8.8479137, 27.1191578, -43.5368805, 56.3465233
3: -18.1541958, 60.4720726, -9.6055470, 34.9412117, -53.0954056, 70.0776215
4: -15.1153126, 56.5998192, -8.5049210, 32.2896156, -47.4049301, 65.1047363

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.1109037, 41.8675880, -8.1963463, 26.3033180, -40.4142189, 50.0639343
1: -19.5679359, 43.4133492, -11.7076044, 27.3492546, -46.9171906, 55.1209526
2: -16.7519665, 48.3994026, -10.0907116, 30.4925594, -47.2445145, 58.4901123
3: -18.5002823, 61.5973701, -10.9913187, 39.2384949, -57.7387772, 72.5886841
4: -15.4016666, 57.6641388, -9.5954742, 36.2742271, -51.6758957, 67.2596130

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.6966248, 52.6518402, -7.1610823, 23.2780495, -40.9746590, 59.8129234
1: -24.6023140, 54.4827576, -10.2265310, 24.2250175, -48.8273277, 64.7092819
2: -21.0324039, 60.5593491, -8.8366385, 27.1080627, -48.1404648, 69.3959885
3: -23.2434731, 77.4685516, -9.6014547, 34.9294739, -58.1729431, 87.0700073
4: -19.2839870, 72.1569519, -8.4956789, 32.2707710, -51.5547562, 80.6526260

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.9450722, 53.3562126, -8.0377760, 25.9523048, -43.8973770, 61.3939819
1: -24.9718819, 55.2171478, -11.4928379, 26.9805431, -51.9524231, 66.7099838
2: -21.3408375, 61.3745003, -9.8984032, 30.0813789, -51.4222183, 71.2729034
3: -23.5634995, 78.4572372, -10.8128338, 38.7355080, -62.2990074, 89.2700729
4: -19.5413036, 73.1064529, -9.4328480, 35.7777252, -55.3190193, 82.5392990

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.1109037, 41.8675880, -14.0264816, 41.5607224, -55.6716232, 55.8940697
1: -19.5679359, 43.4133492, -19.4275627, 43.0964813, -62.6644173, 62.8409119
2: -16.7519665, 48.3994026, -16.6255379, 48.0640945, -64.8160400, 65.0249405
3: -18.5002823, 61.5973701, -18.3753910, 61.1247749, -79.6250534, 79.9727554
4: -15.4016666, 57.6641388, -15.2870178, 57.2594147, -72.6610794, 72.9511566

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.1109037, 41.8675880, -17.7583504, 52.7161827, -66.8270798, 59.6259308
1: -19.5679359, 43.4133492, -24.6639824, 54.5493889, -74.1173248, 68.0773163
2: -16.7519665, 48.3994026, -21.0661926, 60.6673622, -77.4193115, 69.4655914
3: -18.5002823, 61.5973701, -23.2926273, 77.4969101, -95.9971848, 84.8899994
4: -15.4016666, 57.6641388, -19.3064880, 72.2635345, -87.6651993, 76.9706268

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096874
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17.9450722, 53.3562126, -14.0264816, 41.5607224, -59.5057945, 67.3826904
1: -24.9718819, 55.2171478, -19.4275627, 43.0964813, -68.0683517, 74.6447144
2: -21.3408375, 61.3745003, -16.6255379, 48.0640945, -69.4049301, 78.0000381
3: -23.5634995, 78.4572372, -18.3753910, 61.1247749, -84.6882782, 96.8326263
4: -19.5413036, 73.1064529, -15.2870178, 57.2594147, -76.8007202, 88.3934708

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185223, upper bound: 60.2297188
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184850, upper bound: 60.2234170
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17.9450722, 53.3562126, -17.7583504, 52.7161827, -70.6612549, 71.1145554
1: -24.9718819, 55.2171478, -24.6639824, 54.5493889, -79.5212708, 79.8811264
2: -21.3408375, 61.3745003, -21.0661926, 60.6673622, -82.0082016, 82.4406891
3: -23.5634995, 78.4572372, -23.2926273, 77.4969101, -101.0604095, 101.7498627
4: -19.5413036, 73.1064529, -19.3064880, 72.2635345, -91.8048325, 92.4129410

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185223, upper bound: 60.2353227
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184850, upper bound: 60.2280924
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.9862747, 25.6116962, -9.0973234, 29.0640278, -37.0503006, 34.7090187
1: -11.4230928, 26.6387024, -12.8702602, 30.1397648, -41.5628586, 39.5089645
2: -9.8432503, 29.7230167, -11.0743856, 33.6567726, -43.5000229, 40.7974014
3: -10.7079391, 38.2268753, -12.1959352, 43.2404213, -53.9483566, 50.4228096
4: -9.3737078, 35.3655739, -10.4772635, 40.0022278, -49.3759308, 45.8428345

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174484, upper bound: 60.2253227
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2177256, upper bound: 60.2246050
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.9372950, 25.4915276, -10.8696327, 34.4083939, -42.3456879, 36.3611603
1: -11.3542061, 26.5130882, -15.3343544, 35.6384964, -46.9927025, 41.8474426
2: -9.7826948, 29.5849953, -13.1785946, 39.7972412, -49.5799370, 42.7635880
3: -10.6497316, 38.0536957, -14.4879837, 51.1623955, -61.8121223, 52.5416756
4: -9.3202324, 35.2006721, -12.3919563, 47.3149300, -56.6351624, 47.5926285

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2179585, upper bound: 60.2271632
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2204278, upper bound: 60.2275052
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2177256, upper bound: 60.2258507
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.4350300, 29.5662556, -9.0477381, 28.9182739, -38.3532944, 38.6139946
1: -13.3337088, 30.6914234, -12.8007927, 29.9888134, -43.3225212, 43.4922180
2: -11.4997158, 34.2629051, -11.0153551, 33.4909515, -44.9906693, 45.2782593
3: -12.5133295, 44.0571518, -12.1309681, 43.0274811, -55.5408096, 56.1881180
4: -10.8592606, 40.8404045, -10.4249249, 39.8053703, -50.6646309, 51.2653275

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2080387, upper bound: 60.2113171
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2177256, upper bound: 60.2227519
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.3806534, 29.4300499, -10.8205080, 34.2633667, -43.6440201, 40.2505455
1: -13.2568045, 30.5490322, -15.2658110, 35.4886589, -48.7454567, 45.8148384
2: -11.4320164, 34.1069641, -13.1199341, 39.6327248, -51.0647354, 47.2268982
3: -12.4479713, 43.8590813, -14.4236307, 50.9518318, -63.3997917, 58.2827034
4: -10.7990856, 40.6537170, -12.3401222, 47.1205635, -57.9196472, 52.9938354

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110181, upper bound: 60.2125535
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207050, upper bound: 60.2239883
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.7608399, 24.9325790, -13.3044767, 40.1446075, -47.9054489, 38.2370567
1: -11.1007547, 25.9387341, -18.4690800, 41.5440445, -52.6447983, 44.4078064
2: -9.5673141, 28.9531212, -15.8064594, 46.3984032, -55.9657173, 44.7595825
3: -10.4082117, 37.2258720, -17.5225220, 59.2194366, -69.6276474, 54.7483864
4: -9.1240196, 34.4484978, -14.6301203, 55.2837143, -64.4077301, 49.0786171

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148245, upper bound: 60.2115563
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148245, upper bound: 60.2203534
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.6332655, 24.6776619, -16.8450050, 51.1080399, -58.7413063, 41.5226593
1: -10.9307442, 25.6706600, -23.4218330, 52.8250275, -63.7557716, 49.0924911
2: -9.4134912, 28.6531925, -20.0210838, 58.7532997, -68.1667862, 48.6742783
3: -10.2734804, 36.8665810, -22.2593422, 75.3798065, -85.6532822, 59.1259232
4: -8.9967575, 34.0851936, -18.5738354, 69.9567947, -78.9535522, 52.6590271

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2271083, upper bound: 60.2191918
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2275828, upper bound: 60.2201971
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2267058, upper bound: 60.2152353
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.2151823, 28.8945484, -13.2425547, 39.9563522, -49.1715317, 42.1371040
1: -13.0186195, 30.0003948, -18.3801975, 41.3501358, -54.3687553, 48.3805923
2: -11.2287579, 33.5028915, -15.7314644, 46.1865578, -57.4153099, 49.2343521
3: -12.2237511, 43.0676422, -17.4421902, 58.9499397, -71.1736832, 60.5098305
4: -10.6142921, 39.9350967, -14.5643787, 55.0331383, -65.6474228, 54.4994736

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2143397, upper bound: 60.2109232
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2143397, upper bound: 60.2134263
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.0617628, 28.5582962, -16.7918472, 50.9487610, -60.0105133, 45.3501434
1: -12.8152218, 29.6510944, -23.3461895, 52.6602211, -65.4754410, 52.9972839
2: -11.0508614, 33.1085434, -19.9585514, 58.5739212, -69.6247864, 53.0670929
3: -12.0448761, 42.5759201, -22.1896610, 75.1499557, -87.1948242, 64.7655792
4: -10.4630604, 39.4518051, -18.5179958, 69.7442703, -80.2073288, 57.9697952

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2249681, upper bound: 60.2112594
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2249681, upper bound: 60.2137625
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.8627434, 44.4193230, -8.9127254, 28.5410595, -43.4037971, 53.3320465
1: -20.4770241, 46.0121498, -12.6067314, 29.6007442, -50.0777664, 58.6188736
2: -17.6456852, 51.2711029, -10.8521481, 33.0609627, -50.7066498, 62.1232529
3: -19.5267849, 65.8024368, -11.9536343, 42.4839859, -62.0107651, 77.7560730
4: -16.3723183, 61.1303329, -10.2816601, 39.2907829, -55.6630974, 71.4119949

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2057354, upper bound: 60.2180127
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2135910, upper bound: 60.2287571
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.8154125, 44.3104019, -10.6968594, 33.9088364, -48.7242470, 55.0072556
1: -20.4098644, 45.8973656, -15.0916224, 35.1240158, -55.5338821, 60.9889793
2: -17.5824509, 51.1451225, -12.9737864, 39.2294922, -56.8119431, 64.1189117
3: -19.4755459, 65.6449280, -14.2583523, 50.4437637, -69.9193039, 79.9032822
4: -16.3237705, 60.9802094, -12.2118120, 46.6417999, -62.9655685, 73.1920242

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2179366, upper bound: 60.2301545
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167937, upper bound: 60.2306229
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.3341827, 48.6779442, -9.0973234, 29.0640278, -45.3982086, 57.7752686
1: -22.6960735, 50.4090309, -12.8702602, 30.1397648, -52.8358345, 63.2792816
2: -19.4665813, 56.0498695, -11.0743856, 33.6567726, -53.1233521, 67.1242523
3: -21.4443798, 71.8867035, -12.1959352, 43.2404213, -64.6847916, 84.0826416
4: -17.9188328, 66.8347778, -10.4772635, 40.0022278, -57.9210587, 77.3120346

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1982588, upper bound: 60.2090700
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167233, upper bound: 60.2256760
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174617, upper bound: 60.2262974
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.2770634, 48.5410461, -10.8696327, 34.4083939, -50.6854553, 59.4106789
1: -22.6113091, 50.2662735, -15.3343544, 35.6384964, -58.2497978, 65.6006165
2: -19.3896599, 55.8921509, -13.1785946, 39.7972412, -59.1868973, 69.0707397
3: -21.3798370, 71.6921005, -14.4879837, 51.1623955, -72.5422363, 86.1800842
4: -17.8592091, 66.6471710, -12.3919563, 47.3149300, -65.1741409, 79.0391159

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197052, upper bound: 60.2279134
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174617, upper bound: 60.2278335
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.8627434, 44.4193230, -14.6242218, 44.3043480, -59.1670876, 59.0435448
1: -20.4770241, 46.0121498, -20.2457829, 45.8350677, -66.3120880, 66.2579117
2: -17.6456852, 51.2711029, -17.4124851, 51.0891838, -68.7348709, 68.6835709
3: -19.5267849, 65.8024368, -19.2638702, 65.6291351, -85.1559143, 85.0663071
4: -16.3723183, 61.1303329, -16.2089767, 60.8570938, -77.2294159, 77.3393097

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2149672, upper bound: 60.2217008
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153014, upper bound: 60.2211567
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.8627434, 44.4193230, -15.8565083, 47.9964638, -62.8591995, 60.2758331
1: -20.4770241, 46.0121498, -22.0979061, 49.6354141, -70.1124420, 68.1100388
2: -17.6456852, 51.2711029, -18.9476433, 55.2243576, -72.8700409, 70.2187347
3: -19.5267849, 65.8024368, -20.9038620, 70.9059448, -90.4327087, 86.7062988
4: -16.3723183, 61.1303329, -17.5394630, 65.7744293, -82.1467438, 78.6697998

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2149672, upper bound: 60.2306813
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153014, upper bound: 60.2310668
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16.3341827, 48.6779442, -15.6605654, 47.3928490, -63.7270317, 64.3385086
1: -22.6960735, 50.4090309, -21.8332367, 49.0203247, -71.7164001, 72.2422638
2: -19.4665813, 56.0498695, -18.7182560, 54.5562439, -74.0228119, 74.7681198
3: -21.4443798, 71.8867035, -20.6293983, 70.0185394, -91.4629059, 92.5160904
4: -17.9188328, 66.8347778, -17.3239307, 64.9725037, -82.8913193, 84.1586990

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2188254, upper bound: 60.2250373
time: 1.13 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2188254, upper bound: 60.2250373
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.3341827, 48.6779442, -15.0755749, 45.9713211, -62.3055038, 63.7535172
1: -22.6960735, 50.4090309, -21.0839806, 47.5322456, -70.2283173, 71.4930115
2: -19.4665813, 56.0498695, -18.0686455, 52.9041939, -72.3707657, 74.1185150
3: -21.4443798, 71.8867035, -19.9506207, 68.0000534, -89.4444275, 91.8373260
4: -17.9188328, 66.8347778, -16.7616405, 63.0414658, -80.9602814, 83.5964203

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198138, upper bound: 60.2261004
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198138, upper bound: 60.2261004
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.7370825, 30.8880539, -7.8506246, 24.4222717, -34.1593552, 38.7386742
1: -13.8120041, 32.0481262, -11.2154808, 25.5163822, -39.3283844, 43.2636070
2: -11.8939190, 35.7687950, -9.7223167, 28.4403000, -40.3342209, 45.4911118
3: -13.0009775, 45.9720306, -10.3808765, 36.4236946, -49.4246635, 56.3529053
4: -11.2234163, 42.5065231, -9.2233162, 33.8441010, -45.0675125, 51.7298355

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1946284, upper bound: 60.2000138
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1753187, upper bound: 60.1866901
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.7370825, 30.8880539, -8.5392914, 27.2805252, -37.0176048, 39.4273338
1: -13.8120041, 32.0481262, -12.1856346, 28.3512688, -42.1632729, 44.2337608
2: -11.8939190, 35.7687950, -10.4923859, 31.6131878, -43.5071030, 46.2611809
3: -13.0009775, 45.9720306, -11.4304800, 40.6814766, -53.6824493, 57.4025116
4: -11.2234163, 42.5065231, -9.9563656, 37.6183701, -48.8417854, 52.4628830

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1946284, upper bound: 60.2071259
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1853469, upper bound: 60.1880146
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.7370825, 30.8880539, -9.7370825, 30.8880539, -40.6251259, 40.6251259
1: -13.8120041, 32.0481262, -13.8120041, 32.0481262, -45.8601303, 45.8601303
2: -11.8939190, 35.7687950, -11.8939190, 35.7687950, -47.6627121, 47.6627121
3: -13.0009775, 45.9720306, -13.0009775, 45.9720306, -58.9730072, 58.9730072
4: -11.2234163, 42.5065231, -11.2234163, 42.5065231, -53.7299385, 53.7299385

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1800941, upper bound: 60.2071259
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1711513, upper bound: 60.1880146
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.6605654, 47.3928490, -8.1285038, 26.1255989, -41.7861633, 55.5213547
1: -21.8332367, 49.0203247, -11.6347637, 27.1772423, -49.0104790, 60.6550903
2: -18.7182560, 54.5562439, -10.0322857, 30.3130684, -49.0313263, 64.5885315
3: -20.6293983, 70.0185394, -10.8922729, 38.9926758, -59.6220741, 80.9108124
4: -17.3239307, 64.9725037, -9.5428371, 36.0592842, -53.3832169, 74.5153198

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.6605654, 47.3928490, -8.5153341, 27.3904495, -43.0510139, 55.9081802
1: -21.8332367, 49.0203247, -12.1967220, 28.4602489, -50.2934875, 61.2170486
2: -18.7182560, 54.5562439, -10.4935160, 31.7376595, -50.4559174, 65.0497589
3: -20.6293983, 70.0185394, -11.4414949, 40.8764229, -61.5058212, 81.4600372
4: -17.3239307, 64.9725037, -9.9631777, 37.7654228, -55.0893555, 74.9356842

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.0755749, 45.9713211, -8.1285038, 26.1255989, -41.2011681, 54.0998230
1: -21.0839806, 47.5322456, -11.6347637, 27.1772423, -48.2612228, 59.1670074
2: -18.0686455, 52.9041939, -10.0322857, 30.3130684, -48.3817139, 62.9364700
3: -19.9506207, 68.0000534, -10.8922729, 38.9926758, -58.9432983, 78.8923264
4: -16.7616405, 63.0414658, -9.5428371, 36.0592842, -52.8209229, 72.5842972

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2046576, upper bound: 60.2292441
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2285014
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.0755749, 45.9713211, -8.5153341, 27.3904495, -42.4660263, 54.4866524
1: -21.0839806, 47.5322456, -12.1967220, 28.4602489, -49.5442276, 59.7289658
2: -18.0686455, 52.9041939, -10.4935160, 31.7376595, -49.8063049, 63.3977089
3: -19.9506207, 68.0000534, -11.4414949, 40.8764229, -60.8270416, 79.4415512
4: -16.7616405, 63.0414658, -9.9631777, 37.7654228, -54.5270615, 73.0046463

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2046576, upper bound: 60.2292441
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2285014
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.6605654, 47.3928490, -9.4902773, 30.2788124, -45.9393768, 56.8831253
1: -21.8332367, 49.0203247, -13.4819546, 31.4256840, -53.2589188, 62.5022812
2: -18.7182560, 54.5562439, -11.6121740, 35.0876083, -53.8058624, 66.1684113
3: -20.6293983, 70.0185394, -12.6993761, 45.0885582, -65.7179489, 82.7179184
4: -17.3239307, 64.9725037, -10.9753227, 41.6790657, -59.0029984, 75.9478149

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.6605654, 47.3928490, -8.8527384, 28.6118259, -44.2723885, 56.2455788
1: -21.8332367, 49.0203247, -12.6701527, 29.6863174, -51.5195541, 61.6904755
2: -18.7182560, 54.5562439, -10.9015503, 33.1424103, -51.8606644, 65.4577942
3: -20.6293983, 70.0185394, -11.9077425, 42.7056770, -63.3350716, 81.9262772
4: -17.3239307, 64.9725037, -10.3361197, 39.4079056, -56.7318344, 75.3086243

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.0755749, 45.9713211, -9.4902773, 30.2788124, -45.3543854, 55.4615974
1: -21.0839806, 47.5322456, -13.4819546, 31.4256840, -52.5096626, 61.0141983
2: -18.0686455, 52.9041939, -11.6121740, 35.0876083, -53.1562538, 64.5163651
3: -19.9506207, 68.0000534, -12.6993761, 45.0885582, -65.0391769, 80.6994324
4: -16.7616405, 63.0414658, -10.9753227, 41.6790657, -58.4407043, 74.0167847

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2022976, upper bound: 60.2273034
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183178, upper bound: 60.2265701
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.0755749, 45.9713211, -8.8527384, 28.6118259, -43.6873970, 54.8240509
1: -21.0839806, 47.5322456, -12.6701527, 29.6863174, -50.7702980, 60.2024002
2: -18.0686455, 52.9041939, -10.9015503, 33.1424103, -51.2110558, 63.8057442
3: -19.9506207, 68.0000534, -11.9077425, 42.7056770, -62.6562958, 79.9077988
4: -16.7616405, 63.0414658, -10.3361197, 39.4079056, -56.1695480, 73.3775864

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2022976, upper bound: 60.2283567
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183178, upper bound: 60.2276223
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -16.3351402, 48.6634521, -57.7607765, 45.3991661
1: -12.8702602, 30.1397648, -22.7144012, 50.3948059, -63.2650681, 52.8541641
2: -11.0743856, 33.6567726, -19.4737625, 56.0623779, -67.1367645, 53.1305351
3: -12.1959352, 43.2404213, -21.4415531, 71.8649139, -84.0608521, 64.6819763
4: -10.4772635, 40.0022278, -17.9160118, 66.8591766, -77.3364258, 57.9182358

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2036083, upper bound: 60.1919339
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2036083, upper bound: 60.2169543
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -15.8294182, 47.4725189, -56.5698433, 44.8934402
1: -12.8702602, 30.1397648, -22.0219250, 49.1555405, -62.0257988, 52.1616898
2: -11.0743856, 33.6567726, -18.8785973, 54.6932411, -65.7676239, 52.5353699
3: -12.1959352, 43.2404213, -20.8382092, 70.1864471, -82.3823853, 64.0786285
4: -10.4772635, 40.0022278, -17.4378319, 65.1942596, -75.6715012, 57.4400520

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2252287, upper bound: 60.2140165
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2256395, upper bound: 60.2144482
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -16.2798195, 48.5323944, -59.4020271, 50.6882095
1: -15.3343544, 35.6384964, -22.6321545, 50.2575035, -65.5918579, 58.2706451
2: -13.1785946, 39.7972412, -19.3989067, 55.9111443, -69.0897293, 59.1961365
3: -14.4879837, 51.1623955, -21.3794823, 71.6782990, -86.1662827, 72.5418777
4: -12.3919563, 47.3149300, -17.8582516, 66.6791916, -79.0711441, 65.1731796

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -15.7767868, 47.3498688, -58.2195015, 50.1851807
1: -15.3343544, 35.6384964, -21.9448528, 49.0263023, -64.3606339, 57.5833511
2: -13.1785946, 39.7972412, -18.8082542, 54.5515594, -67.7301483, 58.6054878
3: -14.4879837, 51.1623955, -20.7795296, 70.0108795, -84.4988632, 71.9419250
4: -12.3919563, 47.3149300, -17.3835907, 65.0242157, -77.4161682, 64.6985168

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2269328, upper bound: 60.2165825
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2278958, upper bound: 60.2174301
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -15.8783531, 47.9623566, -57.0596809, 44.9423752
1: -12.8702602, 30.1397648, -22.1452065, 49.6130943, -62.4833527, 52.2849731
2: -11.0743856, 33.6567726, -18.9856815, 55.2107620, -66.2851486, 52.6424561
3: -12.1959352, 43.2404213, -20.9067287, 70.8441391, -83.0400772, 64.1471481
4: -10.4772635, 40.0022278, -17.5473309, 65.7760315, -76.2532883, 57.5495605

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226108, upper bound: 60.2144228
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226108, upper bound: 60.2148587
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -15.3220749, 46.6076012, -55.7049255, 44.3861008
1: -12.8702602, 30.1397648, -21.4345512, 48.1974716, -61.0677338, 51.5743141
2: -11.0743856, 33.6567726, -18.3724213, 53.6409798, -64.7153625, 52.0291939
3: -12.1959352, 43.2404213, -20.2598228, 68.9238205, -81.1197510, 63.5002441
4: -10.4772635, 40.0022278, -17.0110588, 63.9366035, -74.4138489, 57.0132828

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250117, upper bound: 60.2153926
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250117, upper bound: 60.2158284
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -15.8190727, 47.8245544, -58.6941872, 50.2274666
1: -15.3343544, 35.6384964, -22.0594406, 49.4686966, -64.8030472, 57.6979294
2: -13.1785946, 39.7972412, -18.9085503, 55.0527458, -68.2313385, 58.7057915
3: -14.4879837, 51.1623955, -20.8398495, 70.6454773, -85.1334610, 72.0022430
4: -12.3919563, 47.3149300, -17.4869099, 65.5846405, -77.9765778, 64.8018417

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240575, upper bound: 60.2169514
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240575, upper bound: 60.2178423
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -15.2660122, 46.4817200, -57.3513527, 49.6744080
1: -15.3343544, 35.6384964, -21.3538361, 48.0642853, -63.3986397, 56.9923325
2: -13.1785946, 39.7972412, -18.2986755, 53.4946518, -66.6732254, 58.0959167
3: -14.4879837, 51.1623955, -20.1980858, 68.7426376, -83.2306137, 71.3604813
4: -12.3919563, 47.3149300, -16.9536190, 63.7611160, -76.1530533, 64.2685471

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2269885, upper bound: 60.2179212
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250117, upper bound: 60.2188121
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.4011230, 43.6175117, -13.9810228, 41.4775620, -55.8786850, 57.5985298
1: -19.9237652, 45.1279793, -19.3727245, 43.0168571, -62.9406204, 64.5006943
2: -17.1381760, 50.3091011, -16.5899334, 47.9748306, -65.1130066, 66.8990326
3: -18.9676342, 64.6169815, -18.3376999, 61.0801544, -80.0477753, 82.9546738
4: -15.9549217, 59.9325905, -15.2747869, 57.1861076, -73.1410294, 75.2073746

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2010556, upper bound: 60.2178197
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2035032, upper bound: 60.2208159
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.2701521, 43.4083481, -17.8972836, 53.2437401, -67.5138931, 61.3056259
1: -19.7633266, 44.9080849, -24.9060879, 55.1063347, -74.8696594, 69.8141632
2: -17.0046043, 50.0661469, -21.2862034, 61.2662125, -78.2707977, 71.3523483
3: -18.8302383, 64.3283615, -23.5137844, 78.3294296, -97.1596680, 87.8421478
4: -15.8524036, 59.6322556, -19.5124397, 73.0005188, -88.8529053, 79.1446991

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083138, upper bound: 60.2190426
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2107615, upper bound: 60.2222946
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.6426764, 47.3275642, -14.2516823, 42.2935257, -57.9361916, 61.5792427
1: -21.7922020, 48.9486465, -19.7763004, 43.8626213, -65.6548233, 68.7249451
2: -18.6856174, 54.4668770, -16.9289055, 48.8913803, -67.5769958, 71.3957748
3: -20.6187820, 69.9195786, -18.6889801, 62.2258568, -82.8446350, 88.6085587
4: -17.2955723, 64.8848114, -15.5660553, 58.2696075, -75.5651779, 80.4508514

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101621, upper bound: 60.2128238
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101621, upper bound: 60.2194369
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.4533587, 46.9513283, -18.1482563, 53.9570847, -69.4104462, 65.0995712
1: -21.5374165, 48.5503235, -25.2792187, 55.8586731, -77.3960876, 73.8295441
2: -18.4542274, 54.0260010, -21.5976944, 62.0915871, -80.5458145, 75.6236954
3: -20.3998470, 69.3874435, -23.8372631, 79.3302765, -99.7301102, 93.2246933
4: -17.1273994, 64.3239212, -19.7728081, 73.9612427, -91.0886383, 84.0967255

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197028, upper bound: 60.2174890
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197028, upper bound: 60.2241022
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.4011230, 43.6175117, -13.3000851, 40.1295547, -54.5306778, 56.9175949
1: -19.9237652, 45.1279793, -18.4608459, 41.5330505, -61.4568176, 63.5888252
2: -17.1381760, 50.3091011, -15.8081255, 46.3879585, -63.5261345, 66.1172256
3: -18.9676342, 64.6169815, -17.5145912, 59.2097702, -78.1773987, 82.1315613
4: -15.9549217, 59.9325905, -14.6339607, 55.2861137, -71.2410355, 74.5665512

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2004981, upper bound: 60.2170369
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2029457, upper bound: 60.2200167
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.2701521, 43.4083481, -17.3653793, 52.4395332, -66.7096786, 60.7737198
1: -19.7633266, 44.9080849, -24.1630001, 54.2078819, -73.9712067, 69.0710602
2: -17.0046043, 50.0661469, -20.6616344, 60.2733116, -77.2779160, 70.7277756
3: -18.8302383, 64.3283615, -22.9183426, 77.3106232, -96.1408615, 87.2467041
4: -15.8524036, 59.6322556, -19.0929070, 71.8018799, -87.6542740, 78.7251587

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1868096, upper bound: 60.1868348
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1868098, upper bound: 60.1875534
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.8565083, 47.9964638, -14.7999096, 44.7783546, -60.6348648, 62.7963715
1: -22.0979061, 49.6354141, -20.4935341, 46.3249893, -68.4228973, 70.1289520
2: -18.9476433, 55.2243576, -17.6280575, 51.6312027, -70.5788422, 72.8524170
3: -20.9038620, 70.9059448, -19.4902897, 66.3224258, -87.2262802, 90.3962173
4: -17.5394630, 65.7744293, -16.3931046, 61.5164261, -79.0558929, 82.1675339

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2025053, upper bound: 60.2000943
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2116106, upper bound: 60.2116103
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.8565083, 47.9964638, -16.1026459, 48.6440735, -64.5005798, 64.0991058
1: -22.0979061, 49.6354141, -22.4510231, 50.3103485, -72.4082489, 72.0864410
2: -18.9476433, 55.2243576, -19.2496815, 55.9721222, -74.9197693, 74.4740372
3: -20.9038620, 70.9059448, -21.2180367, 71.8491440, -92.7529984, 92.1239777
4: -17.5394630, 65.7744293, -17.7955246, 66.6930542, -84.2325058, 83.5699539

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2025053, upper bound: 60.2091662
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2116106, upper bound: 60.2125939
time: 0.92 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.92 seconds
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2291491, upper bound: 60.2338100
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2292948, upper bound: 60.2335783
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2326945, upper bound: 60.2246799
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2334013, upper bound: 60.2319485
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2326945, upper bound: 60.2261504
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2334013, upper bound: 60.2319485
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2154364, upper bound: 60.2221333
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2152922, upper bound: 60.2222616
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2250757, upper bound: 60.2288231
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2248943, upper bound: 60.2280087
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2232131
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2162423, upper bound: 60.2166351
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2261480
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2162423, upper bound: 60.2194586
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096874
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2185223, upper bound: 60.2297188
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2184850, upper bound: 60.2234170
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2185223, upper bound: 60.2353227
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2184850, upper bound: 60.2280924
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2174484, upper bound: 60.2253227
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2177256, upper bound: 60.2246050
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2204278, upper bound: 60.2275052
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2177256, upper bound: 60.2258507
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2080387, upper bound: 60.2113171
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2177256, upper bound: 60.2227519
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2110181, upper bound: 60.2125535
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2207050, upper bound: 60.2239883
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2148245, upper bound: 60.2115563
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2148245, upper bound: 60.2203534
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2275828, upper bound: 60.2201971
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2267058, upper bound: 60.2152353
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2143397, upper bound: 60.2109232
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2143397, upper bound: 60.2134263
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2249681, upper bound: 60.2112594
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2249681, upper bound: 60.2137625
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2057354, upper bound: 60.2180127
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2135910, upper bound: 60.2287571
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2179366, upper bound: 60.2301545
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2167937, upper bound: 60.2306229
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2167233, upper bound: 60.2256760
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2174617, upper bound: 60.2262974
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2197052, upper bound: 60.2279134
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2174617, upper bound: 60.2278335
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2149672, upper bound: 60.2217008
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2153014, upper bound: 60.2211567
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2149672, upper bound: 60.2306813
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2153014, upper bound: 60.2310668
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2188254, upper bound: 60.2250373
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2188254, upper bound: 60.2250373
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2198138, upper bound: 60.2261004
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2198138, upper bound: 60.2261004
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1946284, upper bound: 60.2000138
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1753187, upper bound: 60.1866901
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1946284, upper bound: 60.2071259
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1853469, upper bound: 60.1880146
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1800941, upper bound: 60.2071259
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1711513, upper bound: 60.1880146
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2046576, upper bound: 60.2292441
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2285014
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2046576, upper bound: 60.2292441
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2206779, upper bound: 60.2285014
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2022976, upper bound: 60.2273034
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2183178, upper bound: 60.2265701
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2022976, upper bound: 60.2283567
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2183178, upper bound: 60.2276223
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2036083, upper bound: 60.1919339
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2036083, upper bound: 60.2169543
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2252287, upper bound: 60.2140165
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2256395, upper bound: 60.2144482
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2269328, upper bound: 60.2165825
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2278958, upper bound: 60.2174301
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2226108, upper bound: 60.2144228
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2226108, upper bound: 60.2148587
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2250117, upper bound: 60.2153926
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2250117, upper bound: 60.2158284
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2240575, upper bound: 60.2169514
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2240575, upper bound: 60.2178423
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2269885, upper bound: 60.2179212
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2250117, upper bound: 60.2188121
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2010556, upper bound: 60.2178197
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2035032, upper bound: 60.2208159
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2083138, upper bound: 60.2190426
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2107615, upper bound: 60.2222946
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2101621, upper bound: 60.2128238
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2101621, upper bound: 60.2194369
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2197028, upper bound: 60.2174890
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2197028, upper bound: 60.2241022
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2004981, upper bound: 60.2170369
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2029457, upper bound: 60.2200167
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1868096, upper bound: 60.1868348
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.1868098, upper bound: 60.1875534
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2025053, upper bound: 60.2000943
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2116106, upper bound: 60.2116103
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2025053, upper bound: 60.2091662
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 4, lower bound: -60.2116106, upper bound: 60.2125939

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.5975094, 21.5420189, -8.3547344, 26.7673855, -33.3648834, 29.8967533
1: -9.3674479, 22.4186649, -11.9245319, 27.8209152, -37.1883621, 34.3431969
2: -8.1068649, 25.0993271, -10.2704678, 31.0293274, -39.1361885, 35.3697968
3: -8.8570242, 32.3224869, -11.1917734, 39.9468613, -48.8038864, 43.5142593
4: -7.8217878, 29.8806133, -9.7622013, 36.9261665, -44.7479553, 39.6428146

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291491, upper bound: 60.2312781
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291491, upper bound: 60.2338100
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.6461124, 27.4508648, -8.3104696, 26.6600475, -35.3061600, 35.7613335
1: -12.1212921, 28.5366573, -11.8620310, 27.7084217, -39.8297043, 40.3986893
2: -10.4637842, 31.9594040, -10.2152567, 30.9059868, -41.3697662, 42.1746597
3: -11.4808617, 41.0608101, -11.1398411, 39.7913818, -51.2722397, 52.2006531
4: -9.9898520, 38.0416107, -9.7130795, 36.7780418, -46.7678909, 47.7546921

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2292948, upper bound: 60.2310359
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2292948, upper bound: 60.2335783
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.5930567, 23.7586212, -7.4733706, 24.1488380, -31.7418900, 31.2319908
1: -10.8591785, 24.8284836, -10.6540947, 25.1270370, -35.9862137, 35.4825783
2: -9.4190245, 27.6722012, -9.2093201, 28.1137600, -37.5327835, 36.8815193
3: -10.0636024, 35.4527893, -9.9912634, 36.2262306, -46.2898254, 45.4440536
4: -8.9545259, 32.9169846, -8.8301640, 33.4773788, -42.4318962, 41.7471390

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325023, upper bound: 60.2232579
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.3008766, 26.6333847, -7.4733706, 24.1488380, -32.4497147, 34.1067505
1: -11.8607264, 27.6855011, -10.6540947, 25.1270370, -36.9877625, 38.3395882
2: -10.2210770, 30.8583946, -9.2093201, 28.1137600, -38.3348389, 40.0677109
3: -11.1320915, 39.7278976, -9.9912634, 36.2262306, -47.3583221, 49.7191620
4: -9.7145023, 36.7118607, -8.8301640, 33.4773788, -43.1918755, 45.5420227

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2343116, upper bound: 60.2274021
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291033, upper bound: 60.2219682
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2332239, upper bound: 60.2275423
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2338100, upper bound: 60.2291491
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2335783, upper bound: 60.2292948
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.5930567, 23.7586212, -8.4589252, 27.0919533, -34.6850090, 32.2175446
1: -10.8591785, 24.8284836, -12.0820923, 28.1615906, -39.0207672, 36.9105759
2: -9.4190245, 27.6722012, -10.4120913, 31.3883038, -40.8073273, 38.0842896
3: -10.0636024, 35.4527893, -11.3392544, 40.4026985, -50.4662971, 46.7920418
4: -8.9545259, 32.9169846, -9.8867702, 37.3420067, -46.2965240, 42.8037491

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166061, upper bound: 60.2147709
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2254614, upper bound: 60.2179149
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.3008766, 26.6333847, -8.4589252, 27.0919533, -35.3928299, 35.0923080
1: -11.8607264, 27.6855011, -12.0820923, 28.1615906, -40.0223160, 39.7675858
2: -10.2210770, 30.8583946, -10.4120913, 31.3883038, -41.6093826, 41.2704849
3: -11.1320915, 39.7278976, -11.3392544, 40.4026985, -51.5347900, 51.0671463
4: -9.7145023, 36.7118607, -9.8867702, 37.3420067, -47.0565033, 46.5986328

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2328667, upper bound: 60.2288307
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172534, upper bound: 60.2164181
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281660, upper bound: 60.2281661
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.3862433, 20.9005814, -13.7637329, 40.7737389, -47.1599808, 34.6643143
1: -9.0621185, 21.7562218, -19.0361233, 42.2805214, -51.3426399, 40.7923431
2: -7.8449283, 24.3677864, -16.2971249, 47.1798096, -55.0247383, 40.6649094
3: -8.5754585, 31.3739891, -18.0353546, 60.0240097, -68.5994492, 49.4093437
4: -7.5874710, 29.0066986, -15.0064383, 56.2157898, -63.8032608, 44.0131378

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.4249496, 26.7823811, -13.7116365, 40.6472626, -49.0722122, 40.4940186
1: -11.8032913, 27.8461151, -18.9586468, 42.1468735, -53.9501648, 46.8047600
2: -10.1881123, 31.1995201, -16.2250805, 47.0341530, -57.2222672, 47.4245987
3: -11.1917057, 40.0699387, -17.9773998, 59.8478432, -71.0395279, 58.0473404
4: -9.7411613, 37.1346054, -14.9450293, 56.0435524, -65.7846985, 52.0796356

Time for backsubstitution: 0.83 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.04 + 417.19 = 420.23 seconds
