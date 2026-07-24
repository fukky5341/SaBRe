## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 57.280903066


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068)
1: (-16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383)
2: (-16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280)
3: (-27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894)
4: (-25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.58 = 3.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -57.5687468, upper bound: 57.5687468

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5532159, upper bound: 57.5570901
time: 0.45 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5621985, upper bound: 57.5621985
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.07 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 0, lower bound: -57.5532159, upper bound: 57.5570901
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 0, lower bound: -57.5621985, upper bound: 57.5621985

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.1770201, 42.4380188, -11.8764477, 48.9575691, -59.1345901, 54.3144684
1: -12.9492474, 48.0443954, -15.0760498, 55.3899689, -68.3392181, 63.1204453
2: -12.7204800, 47.7395439, -14.7838812, 55.2434998, -67.9639816, 62.5234261
3: -21.9290562, 51.2056084, -25.3967628, 58.8714714, -80.8005219, 76.6023712
4: -20.4464874, 49.1660233, -23.6019707, 56.9266968, -77.3731613, 72.7679749

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4861621, upper bound: 57.4762997
time: 0.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4861621, upper bound: 57.5570901
time: 0.52 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -13.0451412, 53.5286636, -65.5080032, 62.5160027
1: -15.2106190, 55.9559326, -16.5472050, 60.5470352, -75.7576523, 72.5031357
2: -14.9106045, 55.8369446, -16.2069473, 60.5032959, -75.4138870, 72.0438919
3: -25.6317139, 59.4555740, -27.8260193, 64.2862701, -89.9179688, 87.2815933
4: -23.7928352, 57.5261345, -25.8074226, 62.3490105, -86.1418457, 83.3335419

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
time: 0.43 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621985
time: 0.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -57.4861621, upper bound: 57.4762997
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -57.4861621, upper bound: 57.5570901
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621985

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.6038160, 40.2809982, -10.9257622, 45.6060066, -55.2098198, 51.2067528
1: -12.2323484, 45.6113548, -13.9345284, 51.6139565, -63.8463058, 59.5458755
2: -12.0197229, 45.2531128, -13.5783024, 51.4651146, -63.4848328, 58.8314133
3: -20.7653065, 48.6373138, -23.5746078, 54.8061142, -75.5714035, 72.2119141
4: -19.3651485, 46.5791321, -21.6680050, 52.9406357, -72.3057785, 68.2471390

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4744205
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4762997
time: 0.48 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.1770201, 42.4380188, -10.2766151, 42.9209518, -53.0979729, 52.7146339
1: -12.9492474, 48.0443954, -13.0752325, 48.5794907, -61.5287399, 61.1196251
2: -12.7204800, 47.7395439, -12.8335724, 48.3097000, -61.0301819, 60.5730934
3: -21.9290562, 51.2056084, -22.1424255, 51.7142334, -73.6432877, 73.3480377
4: -20.4464874, 49.1660233, -20.5802517, 49.7248878, -70.1713638, 69.7462692

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5210888
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5570901
time: 0.51 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -10.1770201, 42.4380188, -54.4173622, 59.6478844
1: -15.2106190, 55.9559326, -12.9492474, 48.0443954, -63.2550125, 68.9051743
2: -14.9106045, 55.8369446, -12.7204800, 47.7395439, -62.6501465, 68.5574265
3: -25.6317139, 59.4555740, -21.9290562, 51.2056084, -76.8373108, 81.3846283
4: -23.7928352, 57.5261345, -20.4464874, 49.1660233, -72.9588547, 77.9725952

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4762997, upper bound: 57.4861621
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
time: 0.65 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -11.9793425, 49.4708633, -61.4502068, 61.4502068
1: -15.2106190, 55.9559326, -15.2106190, 55.9559326, -71.1665497, 71.1665497
2: -14.9106045, 55.8369446, -14.9106045, 55.8369446, -70.7475510, 70.7475510
3: -25.6317139, 59.4555740, -25.6317139, 59.4555740, -85.0872879, 85.0872879
4: -23.7928352, 57.5261345, -23.7928352, 57.5261345, -81.3189545, 81.3189621

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4762997, upper bound: 57.4861621
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621587
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.61 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4744205
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4762997
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5210888
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5570901
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.4762997, upper bound: 57.4861621
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.4762997, upper bound: 57.4861621
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621587

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -10.9257622, 45.6060066, -54.7635956, 49.8001862
1: -11.6959438, 44.0212021, -13.9345284, 51.6139565, -63.3098984, 57.9557304
2: -11.4683819, 43.6604576, -13.5783024, 51.4651146, -62.9334946, 57.2387581
3: -19.9659348, 46.8748016, -23.5746078, 54.8061142, -74.7720337, 70.4494095
4: -18.4506874, 44.9865723, -21.6680050, 52.9406357, -71.3913269, 66.6545792

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4744205
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4744205
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -10.9257622, 45.6060066, -54.4178276, 48.2835579
1: -11.2442751, 42.3229713, -13.9345284, 51.6139565, -62.8582306, 56.2574997
2: -11.0679474, 41.8824844, -13.5783024, 51.4651146, -62.5330582, 55.4607849
3: -19.1606712, 45.1869125, -23.5746078, 54.8061142, -73.9667816, 68.7615204
4: -17.8871288, 43.1344337, -21.6680050, 52.9406357, -70.8277664, 64.8024368

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4762997
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4762997
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -10.2766151, 42.9209518, -52.0785446, 49.1510429
1: -11.6959438, 44.0212021, -13.0752325, 48.5794907, -60.2754364, 57.0964317
2: -11.4683819, 43.6604576, -12.8335724, 48.3097000, -59.7780838, 56.4940224
3: -19.9659348, 46.8748016, -22.1424255, 51.7142334, -71.6801605, 69.0172272
4: -18.4506874, 44.9865723, -20.5802517, 49.7248878, -68.1755753, 65.5668259

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5210888
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5210888
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -10.2766151, 42.9209518, -51.7327728, 47.6344109
1: -11.2442751, 42.3229713, -13.0752325, 48.5794907, -59.8237648, 55.3982010
2: -11.0679474, 41.8824844, -12.8335724, 48.3097000, -59.3776474, 54.7160530
3: -19.1606712, 45.1869125, -22.1424255, 51.7142334, -70.8749084, 67.3293381
4: -17.8871288, 43.1344337, -20.5802517, 49.7248878, -67.6120071, 63.7146835

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5512705
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5513040
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -9.6038160, 40.2809982, -51.3107796, 55.7454224
1: -14.0756989, 52.2037125, -12.2323484, 45.6113548, -59.6870537, 64.4360580
2: -13.7119274, 52.0922241, -12.0197229, 45.2531128, -58.9650383, 64.1119461
3: -23.8188839, 55.4197121, -20.7653065, 48.6373138, -72.4561844, 76.1850204
4: -21.8664837, 53.5777702, -19.3651485, 46.5791321, -68.4456177, 72.9429016

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4744205, upper bound: 57.4798383
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4744205, upper bound: 57.4861621
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -10.1770201, 42.4380188, -52.8427162, 53.6730309
1: -13.2431650, 49.2144547, -12.9492474, 48.0443954, -61.2875595, 62.1637039
2: -12.9902172, 48.9800606, -12.7204800, 47.7395439, -60.7297478, 61.7005386
3: -22.4255047, 52.3740425, -21.9290562, 51.2056084, -73.6311111, 74.3030853
4: -20.8130035, 50.4081459, -20.4464874, 49.1660233, -69.9790115, 70.8546066

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5210888, upper bound: 57.5063429
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5210888, upper bound: 57.5532159
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -11.3360462, 47.0514069, -58.0811920, 57.4776535
1: -14.0756989, 52.2037125, -14.4060440, 53.2275314, -67.3032303, 66.6097565
2: -13.7119274, 52.0922241, -14.1231031, 53.0575676, -66.7694931, 66.2153244
3: -23.8188839, 55.4197121, -24.3241272, 56.5804291, -80.3993149, 79.7438354
4: -21.8664837, 53.5777702, -22.5729065, 54.6305351, -76.4970169, 76.1506729

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4669328
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4861621
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -11.9793425, 49.4708633, -59.8755646, 55.4753494
1: -13.2431650, 49.2144547, -15.2106190, 55.9559326, -69.1990967, 64.4250717
2: -12.9902172, 48.9800606, -14.9106045, 55.8369446, -68.8271637, 63.8906631
3: -22.4255047, 52.3740425, -25.6317139, 59.4555740, -81.8810806, 78.0057449
4: -20.8130035, 50.4081459, -23.7928352, 57.5261345, -78.3391266, 74.2009735

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4959195, upper bound: 57.4797424
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4959195, upper bound: 57.5621587
time: 0.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.58 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4744205
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4744205
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4762997
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.4762997
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5210888
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5210888
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5512705
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4798383, upper bound: 57.5513040
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4744205, upper bound: 57.4798383
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4744205, upper bound: 57.4861621
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.5210888, upper bound: 57.5063429
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.5210888, upper bound: 57.5532159
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4669328
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4861621
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4959195, upper bound: 57.4797424
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -57.4959195, upper bound: 57.5621587

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -9.3738651, 39.6521530, -48.8097420, 48.2482910
1: -11.6959438, 44.0212021, -11.9794750, 44.9010925, -56.5970383, 56.0006790
2: -11.4683819, 43.6604576, -11.7082052, 44.5663643, -56.0347443, 55.3686562
3: -19.9659348, 46.8748016, -20.4000530, 47.7824707, -67.7483978, 67.2748566
4: -18.4506874, 44.9865723, -18.8121281, 45.8937721, -64.3444595, 63.7986984

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4452907, upper bound: 57.4004476
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4793540, upper bound: 57.4738035
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -11.0278711, 46.1343651, -55.2919579, 49.9022980
1: -11.6959438, 44.0212021, -14.0733223, 52.1954956, -63.8914413, 58.0945168
2: -11.4683819, 43.6604576, -13.7095766, 52.0840302, -63.5524139, 57.3700333
3: -19.9659348, 46.8748016, -23.8148842, 55.4109116, -75.3768463, 70.6896820
4: -18.4506874, 44.9865723, -21.8627625, 53.5693436, -72.0200348, 66.8493347

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4452907, upper bound: 57.4004476
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4793540, upper bound: 57.4738035
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -9.3738651, 39.6521530, -48.4639740, 46.7316589
1: -11.2442751, 42.3229713, -11.9794750, 44.9010925, -56.1453667, 54.3024445
2: -11.0679474, 41.8824844, -11.7082052, 44.5663643, -55.6343079, 53.5906868
3: -19.1606712, 45.1869125, -20.4000530, 47.7824707, -66.9431458, 65.5869598
4: -17.8871288, 43.1344337, -18.8121281, 45.8937721, -63.7808990, 61.9465637

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4112991, upper bound: 57.2922389
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4858081, upper bound: 57.4758765
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -11.0278711, 46.1343651, -54.9461861, 48.3856659
1: -11.2442751, 42.3229713, -14.0733223, 52.1954956, -63.4397697, 56.3962898
2: -11.0679474, 41.8824844, -13.7095766, 52.0840302, -63.1519775, 55.5920601
3: -19.1606712, 45.1869125, -23.8148842, 55.4109116, -74.5715790, 69.0018005
4: -17.8871288, 43.1344337, -21.8627625, 53.5693436, -71.4564667, 64.9971924

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4112991, upper bound: 57.2922389
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4858081, upper bound: 57.4758765
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -8.7464733, 37.1027641, -46.2603569, 47.6209030
1: -11.6959438, 44.0212021, -11.1623430, 42.0329132, -53.7288589, 55.1835442
2: -11.4683819, 43.6604576, -10.9871893, 41.5945511, -53.0629349, 54.6476364
3: -19.9659348, 46.8748016, -19.0206509, 44.8774719, -64.8433990, 65.8954468
4: -18.4506874, 44.9865723, -17.7598114, 42.8391037, -61.2897911, 62.7463837

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4951600, upper bound: 57.4796354
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5047699, upper bound: 57.5200781
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -10.3980770, 43.4705620, -52.6281509, 49.2725029
1: -11.6959438, 44.0212021, -13.2348881, 49.1855888, -60.8815308, 57.2560883
2: -11.4683819, 43.6604576, -12.9820824, 48.9513512, -60.4197311, 56.6425323
3: -19.9659348, 46.8748016, -22.4113636, 52.3431854, -72.3091049, 69.2861633
4: -18.4506874, 44.9865723, -20.8000755, 50.3786392, -68.8293228, 65.7866516

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4951600, upper bound: 57.4796354
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5047699, upper bound: 57.5200781
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -8.7464733, 37.1027641, -45.9145851, 46.1042709
1: -11.2442751, 42.3229713, -11.1623430, 42.0329132, -53.2771873, 53.4853134
2: -11.0679474, 41.8824844, -10.9871893, 41.5945511, -52.6624908, 52.8696671
3: -19.1606712, 45.1869125, -19.0206509, 44.8774719, -64.0381393, 64.2075577
4: -17.8871288, 43.1344337, -17.7598114, 42.8391037, -60.7262230, 60.8942375

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4730682, upper bound: 57.3933571
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5497019, upper bound: 57.5497019
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -10.3980770, 43.4705620, -52.2823830, 47.7558708
1: -11.2442751, 42.3229713, -13.2348881, 49.1855888, -60.4298630, 55.5578613
2: -11.0679474, 41.8824844, -12.9820824, 48.9513512, -60.0192947, 54.8645630
3: -19.1606712, 45.1869125, -22.4113636, 52.3431854, -71.5038528, 67.5982742
4: -17.8871288, 43.1344337, -20.8000755, 50.3786392, -68.2657623, 63.9345093

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4730682, upper bound: 57.3933571
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5497019, upper bound: 57.5497018
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -9.1575928, 38.8744278, -49.9042130, 55.2991982
1: -14.0756989, 52.2037125, -11.6959438, 44.0212021, -58.0969009, 63.8996582
2: -13.7119274, 52.0922241, -11.4683819, 43.6604576, -57.3723831, 63.5606079
3: -23.8188839, 55.4197121, -19.9659348, 46.8748016, -70.6936874, 75.3856506
4: -21.8664837, 53.5777702, -18.4506874, 44.9865723, -66.8530579, 72.0284576

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1195769, upper bound: 57.2611741
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4738035, upper bound: 57.4793540
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -8.8118200, 37.3577957, -48.3875809, 54.9534302
1: -14.0756989, 52.2037125, -11.2442751, 42.3229713, -56.3986702, 63.4479866
2: -13.7119274, 52.0922241, -11.0679474, 41.8824844, -55.5944099, 63.1601715
3: -23.8188839, 55.4197121, -19.1606712, 45.1869125, -69.0057983, 74.5803833
4: -21.8664837, 53.5777702, -17.8871288, 43.1344337, -65.0009155, 71.4648895

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1195769, upper bound: 57.2852450
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4738035, upper bound: 57.4858081
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -9.1575928, 38.8744278, -49.2791176, 52.6535950
1: -13.2431650, 49.2144547, -11.6959438, 44.0212021, -57.2643661, 60.9104004
2: -12.9902172, 48.9800606, -11.4683819, 43.6604576, -56.6506729, 60.4484406
3: -22.4255047, 52.3740425, -19.9659348, 46.8748016, -69.3003006, 72.3399658
4: -20.8130035, 50.4081459, -18.4506874, 44.9865723, -65.7995758, 68.8588257

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5200781, upper bound: 57.5047699
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -8.8118200, 37.3577957, -47.7624893, 52.3078308
1: -13.2431650, 49.2144547, -11.2442751, 42.3229713, -55.5661354, 60.4587288
2: -12.9902172, 48.9800606, -11.0679474, 41.8824844, -54.8726997, 60.0480080
3: -22.4255047, 52.3740425, -19.1606712, 45.1869125, -67.6124115, 71.5347061
4: -20.8130035, 50.4081459, -17.8871288, 43.1344337, -63.9474335, 68.2952499

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2869449, upper bound: 57.4081359
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5200781, upper bound: 57.5519753
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -11.0297852, 46.1416092, -57.1713943, 57.1713943
1: -14.0756989, 52.2037125, -14.0756989, 52.2037125, -66.2794113, 66.2794113
2: -13.7119274, 52.0922241, -13.7119274, 52.0922241, -65.8041458, 65.8041458
3: -23.8188839, 55.4197121, -23.8188839, 55.4197121, -79.2385941, 79.2385941
4: -21.8664837, 53.5777702, -21.8664837, 53.5777702, -75.4442520, 75.4442520

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0781504, upper bound: 57.1841498
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4665788
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -10.4046993, 43.4960098, -54.5257912, 56.5463028
1: -14.0756989, 52.2037125, -13.2431650, 49.2144547, -63.2901535, 65.4468765
2: -13.7119274, 52.0922241, -12.9902172, 48.9800606, -62.6919861, 65.0824432
3: -23.8188839, 55.4197121, -22.4255047, 52.3740425, -76.1929245, 77.8452148
4: -21.8664837, 53.5777702, -20.8130035, 50.4081459, -72.2746277, 74.3907700

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0781504, upper bound: 57.2726013
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4858081
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -11.0297852, 46.1416092, -56.5463028, 54.5257950
1: -13.2431650, 49.2144547, -14.0756989, 52.2037125, -65.4468765, 63.2901535
2: -12.9902172, 48.9800606, -13.7119274, 52.0922241, -65.0824432, 62.6919861
3: -22.4255047, 52.3740425, -23.8188839, 55.4197121, -77.8452148, 76.1929245
4: -20.8130035, 50.4081459, -21.8664837, 53.5777702, -74.3907700, 72.2746277

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1200631, upper bound: 57.1744923
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4793463
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -10.4046993, 43.4960098, -53.9007034, 53.9006996
1: -13.2431650, 49.2144547, -13.2431650, 49.2144547, -62.4576187, 62.4576187
2: -12.9902172, 48.9800606, -12.9902172, 48.9800606, -61.9702759, 61.9702644
3: -22.4255047, 52.3740425, -22.4255047, 52.3740425, -74.7995453, 74.7995453
4: -20.8130035, 50.4081459, -20.8130035, 50.4081459, -71.2211304, 71.2211304

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1200631, upper bound: 57.3481845
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.5620557
time: 0.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.79 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4452907, upper bound: 57.4004476
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4793540, upper bound: 57.4738035
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4452907, upper bound: 57.4004476
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4793540, upper bound: 57.4738035
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4112991, upper bound: 57.2922389
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4858081, upper bound: 57.4758765
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4112991, upper bound: 57.2922389
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4858081, upper bound: 57.4758765
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4951600, upper bound: 57.4796354
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.5047699, upper bound: 57.5200781
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4951600, upper bound: 57.4796354
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.5047699, upper bound: 57.5200781
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4730682, upper bound: 57.3933571
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.5497019, upper bound: 57.5497019
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4730682, upper bound: 57.3933571
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.5497019, upper bound: 57.5497018
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.1195769, upper bound: 57.2611741
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4738035, upper bound: 57.4793540
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.1195769, upper bound: 57.2852450
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4738035, upper bound: 57.4858081
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.5200781, upper bound: 57.5047699
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.2869449, upper bound: 57.4081359
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.5200781, upper bound: 57.5519753
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.0781504, upper bound: 57.1841498
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4665788
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.0781504, upper bound: 57.2726013
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4858081
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.1200631, upper bound: 57.1744923
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4793463
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.1200631, upper bound: 57.3481845
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -57.4665788, upper bound: 57.5620557

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -8.3131781, 35.6422195, -44.4091454, 46.5121994
1: -11.2745037, 43.2634735, -10.6406422, 40.3789787, -51.6534729, 53.9041138
2: -11.0006275, 42.8983345, -10.4219761, 39.9246559, -50.9252853, 53.3203049
3: -19.5009098, 46.0489769, -18.2333527, 43.0451736, -62.5460815, 64.2823334
4: -17.9422951, 44.1722069, -16.8750362, 41.1079865, -59.0502777, 61.0472412

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4801864, upper bound: 57.4402933
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986063
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -9.3738651, 39.6521530, -48.4924049, 47.1369133
1: -11.2974596, 42.7681274, -11.9794750, 44.9010925, -56.1985512, 54.7476044
2: -11.0862007, 42.3602257, -11.7082052, 44.5663643, -55.6525650, 54.0684280
3: -19.3324356, 45.5596771, -20.4000530, 47.7824707, -67.1148987, 65.9597244
4: -17.8893318, 43.6359253, -18.8121281, 45.8937721, -63.7830963, 62.4480476

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -10.0218754, 42.3258629, -51.0927887, 48.2209015
1: -11.2745037, 43.2634735, -12.8041134, 47.8987923, -59.1732750, 56.0675888
2: -11.0006275, 42.8983345, -12.4860792, 47.6771545, -58.6777802, 55.3844147
3: -19.5009098, 46.0489769, -21.7729702, 50.9077835, -70.4086914, 67.8219452
4: -17.9422951, 44.1722069, -20.0218124, 49.0368729, -66.9791718, 64.1940155

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.6158258, upper bound: 56.4793470
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4451127, upper bound: 57.4002566
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -11.0278711, 46.1343651, -54.9746170, 48.7909203
1: -11.2974596, 42.7681274, -14.0733223, 52.1954956, -63.4929466, 56.8414497
2: -11.0862007, 42.3602257, -13.7095766, 52.0840302, -63.1702309, 56.0698013
3: -19.3324356, 45.5596771, -23.8148842, 55.4109116, -74.7433472, 69.3745575
4: -17.8893318, 43.6359253, -21.8627625, 53.5693436, -71.4586639, 65.4986877

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2611741, upper bound: 57.1195769
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2611741, upper bound: 57.4738035
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -8.3131781, 35.6422195, -43.7068748, 43.5970230
1: -10.3480911, 39.9815979, -10.6406422, 40.3789787, -50.7270699, 50.6222382
2: -10.1572323, 39.4773331, -10.4219761, 39.9246559, -50.0818863, 49.8993034
3: -17.9082813, 42.6824608, -18.2333527, 43.0451736, -60.9534531, 60.9158134
4: -16.6487427, 40.6530380, -16.8750362, 41.1079865, -57.7567291, 57.5280762

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4874880, upper bound: 57.4417602
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5084133, upper bound: 57.5000732
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -9.3738651, 39.6521530, -48.1215286, 45.5573463
1: -10.8127460, 41.0070839, -11.9794750, 44.9010925, -55.7138367, 52.9865570
2: -10.6570034, 40.5084305, -11.7082052, 44.5663643, -55.2233658, 52.2166328
3: -18.4758568, 43.8035660, -20.4000530, 47.7824707, -66.2583313, 64.2036133
4: -17.2884102, 41.7039909, -18.8121281, 45.8937721, -63.1821823, 60.5161209

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5288863, upper bound: 57.5158539
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5288863, upper bound: 57.5158539
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -10.0218754, 42.3258629, -50.3905220, 45.3057175
1: -10.3480911, 39.9815979, -12.8041134, 47.8987923, -58.2468796, 52.7857132
2: -10.1572323, 39.4773331, -12.4860792, 47.6771545, -57.8343849, 51.9634132
3: -17.9082813, 42.6824608, -21.7729702, 50.9077835, -68.8160629, 64.4554291
4: -16.6487427, 40.6530380, -20.0218124, 49.0368729, -65.6856155, 60.6748505

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2804985, upper bound: 57.1275650
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2804985, upper bound: 57.2922389
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -11.0278711, 46.1343651, -54.6037407, 47.2113533
1: -10.8127460, 41.0070839, -14.0733223, 52.1954956, -63.0082359, 55.0804062
2: -10.6570034, 40.5084305, -13.7095766, 52.0840302, -62.7410355, 54.2180061
3: -18.4758568, 43.8035660, -23.8148842, 55.4109116, -73.8867645, 67.6184540
4: -17.2884102, 41.7039909, -21.8627625, 53.5693436, -70.8577576, 63.5667534

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2852450, upper bound: 57.1292886
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2852450, upper bound: 57.4758765
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -7.7646637, 33.4416122, -42.2085342, 45.9636879
1: -11.2745037, 43.2634735, -9.9138098, 37.9258804, -49.2003746, 53.1772842
2: -11.0006275, 42.8983345, -9.7962627, 37.3471298, -48.3477554, 52.6945953
3: -19.5009098, 46.0489769, -17.0018635, 40.5734634, -60.0743713, 63.0508385
4: -17.9422951, 44.1722069, -15.9780416, 38.4507637, -56.3930588, 60.1502457

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4906093, upper bound: 57.4702730
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5069508, upper bound: 57.5171265
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -8.7464733, 37.1027641, -45.9430161, 46.5095291
1: -11.2974596, 42.7681274, -11.1623430, 42.0329132, -53.3303719, 53.9304695
2: -11.0862007, 42.3602257, -10.9871893, 41.5945511, -52.6807518, 53.3474083
3: -19.3324356, 45.5596771, -19.0206509, 44.8774719, -64.2099075, 64.5803299
4: -17.8893318, 43.6359253, -17.7598114, 42.8391037, -60.7284203, 61.3957176

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5226232
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5291075
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -9.4292402, 39.8496513, -48.6165771, 47.6282616
1: -11.2745037, 43.2634735, -12.0111113, 45.1026955, -56.3771935, 55.2745857
2: -11.0006275, 42.8983345, -11.8066349, 44.7557869, -55.7564125, 54.7049713
3: -19.5009098, 46.0489769, -20.4434452, 48.0528374, -67.5537491, 66.4924240
4: -17.9422951, 44.1722069, -19.0179596, 46.0551949, -63.9974899, 63.1901550

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4741026, upper bound: 57.4385264
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4794885, upper bound: 57.4598491
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -10.3980770, 43.4705620, -52.3108139, 48.1611252
1: -11.2974596, 42.7681274, -13.2348881, 49.1855888, -60.4830360, 56.0030136
2: -11.0862007, 42.3602257, -12.9820824, 48.9513512, -60.0375519, 55.3423042
3: -19.3324356, 45.5596771, -22.4113636, 52.3431854, -71.6756210, 67.9710388
4: -17.8893318, 43.6359253, -20.8000755, 50.3786392, -68.2679596, 64.4360046

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3892579, upper bound: 57.2869449
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3892579, upper bound: 57.5200781
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -7.7646637, 33.4416122, -41.5062637, 43.0485039
1: -10.3480911, 39.9815979, -9.9138098, 37.9258804, -48.2739716, 49.8954086
2: -10.1572323, 39.4773331, -9.7962627, 37.3471298, -47.5043640, 49.2735977
3: -17.9082813, 42.6824608, -17.0018635, 40.5734634, -58.4817390, 59.6843224
4: -16.6487427, 40.6530380, -15.9780416, 38.4507637, -55.0995026, 56.6310768

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5136952, upper bound: 57.5136952
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -8.7464733, 37.1027641, -45.5721436, 44.9299545
1: -10.8127460, 41.0070839, -11.1623430, 42.0329132, -52.8456573, 52.1694260
2: -10.6570034, 40.5084305, -10.9871893, 41.5945511, -52.2515564, 51.4956131
3: -18.4758568, 43.8035660, -19.0206509, 44.8774719, -63.3533287, 62.8242149
4: -17.2884102, 41.7039909, -17.7598114, 42.8391037, -60.1275139, 59.4638023

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5388440, upper bound: 57.5311192
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5497019
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -9.4292402, 39.8496513, -47.9143066, 44.7130852
1: -10.3480911, 39.9815979, -12.0111113, 45.1026955, -55.4507866, 51.9927101
2: -10.1572323, 39.4773331, -11.8066349, 44.7557869, -54.9130173, 51.2839661
3: -17.9082813, 42.6824608, -20.4434452, 48.0528374, -65.9611130, 63.1259079
4: -16.6487427, 40.6530380, -19.0179596, 46.0551949, -62.7039375, 59.6709938

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4514586, upper bound: 57.3630711
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3833926, upper bound: 57.2646062
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3833926, upper bound: 57.3933571
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -10.3980770, 43.4705620, -51.9399376, 46.5815544
1: -10.8127460, 41.0070839, -13.2348881, 49.1855888, -59.9983253, 54.2419739
2: -10.6570034, 40.5084305, -12.9820824, 48.9513512, -59.6083450, 53.4905090
3: -18.4758568, 43.8035660, -22.4113636, 52.3431854, -70.8190460, 66.2149277
4: -17.2884102, 41.7039909, -20.8000755, 50.3786392, -67.6670532, 62.5040665

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4166448, upper bound: 57.2966869
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4166448, upper bound: 57.5497018
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -9.1575928, 38.8744278, -49.5701714, 54.1041718
1: -13.6549959, 50.8562546, -11.6959438, 44.0212021, -57.6761971, 62.5522003
2: -13.3063459, 50.7025528, -11.4683819, 43.6604576, -56.9668045, 62.1709213
3: -23.1477661, 54.0001144, -19.9659348, 46.8748016, -70.0225677, 73.9660492
4: -21.2595654, 52.1365585, -18.4506874, 44.9865723, -66.2461395, 70.5872498

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4004476, upper bound: 57.4452907
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4004476, upper bound: 57.4793540
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.0847101, 47.3818054, -7.8077531, 33.6109238, -44.6956291, 55.1895599
1: -14.2560978, 53.6242714, -9.9677572, 38.1186066, -52.3746986, 63.5920296
2: -13.7948971, 53.5095329, -9.8495989, 37.5383224, -51.3332138, 63.3591309
3: -24.3539143, 56.8776131, -17.0945892, 40.7787476, -65.1326599, 73.9721985
4: -22.2534161, 54.9756432, -16.0624065, 38.6468353, -60.9002533, 71.0380402

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1275650, upper bound: 57.2804985
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1275650, upper bound: 57.2852450
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -8.8118200, 37.3577957, -48.0535431, 53.7584076
1: -13.6549959, 50.8562546, -11.2442751, 42.3229713, -55.9779625, 62.1005287
2: -13.3063459, 50.7025528, -11.0679474, 41.8824844, -55.1888313, 61.7704811
3: -23.1477661, 54.0001144, -19.1606712, 45.1869125, -68.3346786, 73.1607819
4: -21.2595654, 52.1365585, -17.8871288, 43.1344337, -64.3939972, 70.0236740

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2922389, upper bound: 57.4112991
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2922389, upper bound: 57.4858081
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -8.1156521, 34.9136467, -44.9330254, 51.0100403
1: -12.8247375, 48.5298729, -10.3809509, 39.5601501, -52.3848801, 58.9108238
2: -12.5208769, 48.2823448, -10.1999111, 39.0782242, -51.5990944, 58.4822540
3: -21.9548664, 51.6154327, -17.8298454, 42.2038307, -64.1586914, 69.4452820
4: -20.2769165, 49.6436996, -16.5416336, 40.2576141, -60.5345306, 66.1853333

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -9.1575928, 38.8744278, -48.9116554, 51.3502960
1: -12.7809610, 47.7437553, -11.6959438, 44.0212021, -56.8021622, 59.4396973
2: -12.5463152, 47.4627686, -11.4683819, 43.6604576, -56.2067719, 58.9311523
3: -21.6898041, 50.8244820, -19.9659348, 46.8748016, -68.5645981, 70.7904053
4: -20.1472874, 48.8353882, -18.4506874, 44.9865723, -65.1338577, 67.2860718

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4796354, upper bound: 57.4951600
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4796354, upper bound: 57.5047699
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -7.8077531, 33.6109238, -43.6303024, 50.7021408
1: -12.8247375, 48.5298729, -9.9677572, 38.1186066, -50.9433365, 58.4976311
2: -12.5208769, 48.2823448, -9.8495989, 37.5383224, -50.0591888, 58.1319427
3: -21.9548664, 51.6154327, -17.0945892, 40.7787476, -62.7336121, 68.7100220
4: -20.2769165, 49.6436996, -16.0624065, 38.6468353, -58.9237518, 65.7061081

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4051504
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4081359
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -8.8118200, 37.3577957, -47.3950272, 51.0045280
1: -12.7809610, 47.7437553, -11.2442751, 42.3229713, -55.1039314, 58.9880295
2: -12.5463152, 47.4627686, -11.0679474, 41.8824844, -54.4287987, 58.5307159
3: -21.6898041, 50.8244820, -19.1606712, 45.1869125, -66.8767090, 69.9851532
4: -20.1472874, 48.8353882, -17.8871288, 43.1344337, -63.2817154, 66.7224960

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5270612, upper bound: 57.5254430
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5270612, upper bound: 57.5519753
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -11.0297852, 46.1416092, -56.8373566, 55.9763718
1: -13.6549959, 50.8562546, -14.0756989, 52.2037125, -65.8587036, 64.9319534
2: -13.3063459, 50.7025528, -13.7119274, 52.0922241, -65.3985672, 64.4144440
3: -23.1477661, 54.0001144, -23.8188839, 55.4197121, -78.5674744, 77.8190002
4: -21.2595654, 52.1365585, -21.8664837, 53.5777702, -74.8373337, 74.0030441

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1841498, upper bound: 57.0781504
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1841498, upper bound: 57.4665788
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -10.4046993, 43.4960098, -54.1917572, 55.3512764
1: -13.6549959, 50.8562546, -13.2431650, 49.2144547, -62.8694344, 64.0994186
2: -13.3063459, 50.7025528, -12.9902172, 48.9800606, -62.2864075, 63.6927376
3: -23.1477661, 54.0001144, -22.4255047, 52.3740425, -75.5218048, 76.4256210
4: -21.2595654, 52.1365585, -20.8130035, 50.4081459, -71.6677094, 72.9495544

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1744923, upper bound: 57.1200631
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1744923, upper bound: 57.4858081
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -11.0297852, 46.1416092, -56.1788368, 53.2224922
1: -12.7809610, 47.7437553, -14.0756989, 52.2037125, -64.9846725, 61.8194542
2: -12.5463152, 47.4627686, -13.7119274, 52.0922241, -64.6385422, 61.1746864
3: -21.6898041, 50.8244820, -23.8188839, 55.4197121, -77.1095123, 74.6433640
4: -20.1472874, 48.8353882, -21.8664837, 53.5777702, -73.7250519, 70.7018738

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2726013, upper bound: 57.1228428
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2726013, upper bound: 57.4793463
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -9.4292402, 39.8496513, -49.8690300, 52.3236275
1: -12.8247375, 48.5298729, -12.0111113, 45.1026955, -57.9274330, 60.5409851
2: -12.5208769, 48.2823448, -11.8066349, 44.7557869, -57.2766571, 60.0889816
3: -21.9548664, 51.6154327, -20.4434452, 48.0528374, -70.0076904, 72.0588760
4: -20.2769165, 49.6436996, -19.0179596, 46.0551949, -66.3321075, 68.6616592

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3481845
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -10.4046993, 43.4960098, -53.5332336, 52.5974007
1: -12.7809610, 47.7437553, -13.2431650, 49.2144547, -61.9954071, 60.9869194
2: -12.5463152, 47.4627686, -12.9902172, 48.9800606, -61.5263748, 60.4529800
3: -21.6898041, 50.8244820, -22.4255047, 52.3740425, -74.0638428, 73.2499847
4: -20.1472874, 48.8353882, -20.8130035, 50.4081459, -70.5554123, 69.6483841

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4156643, upper bound: 57.2959175
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4156643, upper bound: 57.5620557
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.04 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4801864, upper bound: 57.4402933
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986063
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -56.6158258, upper bound: 56.4793470
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4451127, upper bound: 57.4002566
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2611741, upper bound: 57.1195769
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2611741, upper bound: 57.4738035
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4874880, upper bound: 57.4417602
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5084133, upper bound: 57.5000732
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5288863, upper bound: 57.5158539
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5288863, upper bound: 57.5158539
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2804985, upper bound: 57.1275650
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2804985, upper bound: 57.2922389
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2852450, upper bound: 57.1292886
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2852450, upper bound: 57.4758765
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4906093, upper bound: 57.4702730
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5069508, upper bound: 57.5171265
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5226232
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5291075
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4741026, upper bound: 57.4385264
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4794885, upper bound: 57.4598491
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.3892579, upper bound: 57.2869449
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.3892579, upper bound: 57.5200781
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5136952, upper bound: 57.5136952
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5388440, upper bound: 57.5311192
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5497019
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.3833926, upper bound: 57.2646062
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.3833926, upper bound: 57.3933571
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4166448, upper bound: 57.2966869
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4166448, upper bound: 57.5497018
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4004476, upper bound: 57.4452907
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4004476, upper bound: 57.4793540
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.1275650, upper bound: 57.2804985
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.1275650, upper bound: 57.2852450
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2922389, upper bound: 57.4112991
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2922389, upper bound: 57.4858081
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4796354, upper bound: 57.4951600
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4796354, upper bound: 57.5047699
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4051504
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4081359
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5270612, upper bound: 57.5254430
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.5270612, upper bound: 57.5519753
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.1841498, upper bound: 57.0781504
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.1841498, upper bound: 57.4665788
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.1744923, upper bound: 57.1200631
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.1744923, upper bound: 57.4858081
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2726013, upper bound: 57.1228428
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2726013, upper bound: 57.4793463
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3481845
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4156643, upper bound: 57.2959175
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -57.4156643, upper bound: 57.5620557

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -5.4138098, 25.6408691, -34.4078026, 43.6128349
1: -11.2745037, 43.2634735, -6.8880334, 29.2226677, -40.4971619, 50.1515083
2: -11.0006275, 42.8983345, -6.9775701, 28.3149376, -39.3155670, 49.8759041
3: -19.5009098, 46.0489769, -12.2375631, 31.2964668, -50.7973785, 58.2865372
4: -17.9422951, 44.1722069, -11.7668877, 28.9843082, -46.9266052, 55.9390945

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4200260
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4402933
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.4183712, 33.1447525, -6.4946232, 29.0429840, -36.4613571, 39.6393738
1: -9.5417242, 37.5710144, -8.3029099, 33.0288963, -42.5706177, 45.8739243
2: -9.3741312, 37.0595932, -8.2685080, 32.2582207, -41.6323471, 45.3281021
3: -16.6609154, 40.0830879, -14.3753710, 35.4210663, -52.0819740, 54.4584541
4: -15.4489689, 38.1671028, -13.5885582, 33.1541367, -48.6031036, 51.7556572

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4986064
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -8.9847889, 38.9609985, -47.8012505, 46.7478409
1: -11.2974596, 42.7681274, -11.5671473, 44.1251221, -55.4225807, 54.3352737
2: -11.0862007, 42.3602257, -11.2372074, 43.7923737, -54.8785744, 53.5974274
3: -19.3324356, 45.5596771, -19.9395885, 46.9269409, -66.2593765, 65.4992676
4: -17.8893318, 43.6359253, -18.2945824, 45.0634575, -62.9527893, 61.9304924

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -9.0520763, 38.5367279, -47.3769798, 46.8151245
1: -11.2974596, 42.7681274, -11.5747042, 43.6418037, -54.9392624, 54.3428307
2: -11.0862007, 42.3602257, -11.3216534, 43.2610741, -54.3472748, 53.6818771
3: -19.3324356, 45.5596771, -19.7602921, 46.4578094, -65.7902374, 65.3199615
4: -17.8893318, 43.6359253, -18.2454472, 44.5368423, -62.4261665, 61.8813591

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -9.7917204, 41.4309692, -50.1978951, 47.9907455
1: -11.2745037, 43.2634735, -12.5148449, 46.8862114, -58.1607018, 55.7783203
2: -11.0006275, 42.8983345, -12.2084789, 46.6534081, -57.6540375, 55.1068115
3: -19.5009098, 46.0489769, -21.2993679, 49.8507919, -69.3516998, 67.3483429
4: -17.9422951, 44.1722069, -19.5878906, 48.0006866, -65.9429779, 63.7600975

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3474123, upper bound: 57.3359198
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4227593, upper bound: 57.3625036
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -10.6957531, 44.9465866, -53.7868385, 48.4588013
1: -11.2974596, 42.7681274, -13.6549959, 50.8562546, -62.1537056, 56.4231224
2: -11.0862007, 42.3602257, -13.3063459, 50.7025528, -61.7887383, 55.6665726
3: -19.3324356, 45.5596771, -23.1477661, 54.0001144, -73.3325500, 68.7074432
4: -17.8893318, 43.6359253, -21.2595654, 52.1365585, -70.0258789, 64.8954926

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2172797, upper bound: 57.4566298
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2172797, upper bound: 57.4641336
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -5.4138098, 25.6408691, -33.7055283, 40.6976547
1: -10.3480911, 39.9815979, -6.8880334, 29.2226677, -39.5707588, 46.8696327
2: -10.1572323, 39.4773331, -6.9775701, 28.3149376, -38.4721680, 46.4549026
3: -17.9082813, 42.6824608, -12.2375631, 31.2964668, -49.2047501, 54.9200249
4: -16.6487427, 40.6530380, -11.7668877, 28.9843082, -45.6330490, 52.4199257

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4250266
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4417602
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.2892861, 32.3894920, -6.4946232, 29.0429840, -36.3322678, 38.8841171
1: -9.3475819, 36.7363091, -8.3029099, 33.0288963, -42.3764725, 45.0392189
2: -9.2219372, 36.1250496, -8.2685080, 32.2582207, -41.4801559, 44.3935585
3: -16.2676640, 39.2916412, -14.3753710, 35.4210663, -51.6887283, 53.6670074
4: -15.2287788, 37.2015991, -13.5885582, 33.1541367, -48.3829155, 50.7901573

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4604355, upper bound: 57.5000732
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -8.9847889, 38.9609985, -47.4303780, 45.1682739
1: -10.8127460, 41.0070839, -11.5671473, 44.1251221, -54.9378662, 52.5742302
2: -10.6570034, 40.5084305, -11.2372074, 43.7923737, -54.4493752, 51.7456322
3: -18.4758568, 43.8035660, -19.9395885, 46.9269409, -65.4028015, 63.7431488
4: -17.2884102, 41.7039909, -18.2945824, 45.0634575, -62.3518677, 59.9985733

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4702730, upper bound: 57.4865799
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5167745, upper bound: 57.5017011
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -9.0520763, 38.5367279, -47.0061035, 45.2355537
1: -10.8127460, 41.0070839, -11.5747042, 43.6418037, -54.4545517, 52.5817871
2: -10.6570034, 40.5084305, -11.3216534, 43.2610741, -53.9180756, 51.8300781
3: -18.4758568, 43.8035660, -19.7602921, 46.4578094, -64.9336700, 63.5638466
4: -17.2884102, 41.7039909, -18.2454472, 44.5368423, -61.8252525, 59.9494400

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4702730, upper bound: 57.4865799
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5167745, upper bound: 57.5017011
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -10.6884050, 44.9131813, -52.9778442, 45.9722481
1: -10.3480911, 39.9815979, -13.6456528, 50.8184471, -61.1665268, 53.6272507
2: -10.1572323, 39.4773331, -13.2973566, 50.6655159, -60.8227158, 52.7746887
3: -17.9082813, 42.6824608, -23.1317329, 53.9618683, -71.8701401, 65.8141937
4: -16.6487427, 40.6530380, -21.2456455, 52.0988197, -68.7475586, 61.8986816

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2443625, upper bound: 57.1149556
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2804985, upper bound: 57.2922389
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -11.0243721, 47.1488724, -55.6182518, 47.2078514
1: -10.8127460, 41.0070839, -14.1803865, 53.3598557, -64.1725922, 55.1874695
2: -10.6570034, 40.5084305, -13.7203350, 53.2445908, -63.9015884, 54.2287636
3: -18.4758568, 43.8035660, -24.2248249, 56.5950623, -75.0709229, 68.0283813
4: -17.2884102, 41.7039909, -22.1343040, 54.7041283, -71.9925385, 63.8382950

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2453746, upper bound: 57.0864061
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2825370, upper bound: 57.1263918
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -10.6957531, 44.9465866, -53.4159584, 46.8792305
1: -10.8127460, 41.0070839, -13.6549959, 50.8562546, -61.6689987, 54.6620789
2: -10.6570034, 40.5084305, -13.3063459, 50.7025528, -61.3595390, 53.8147736
3: -18.4758568, 43.8035660, -23.1477661, 54.0001144, -72.4759674, 66.9513321
4: -17.2884102, 41.7039909, -21.2595654, 52.1365585, -69.4249725, 62.9635544

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2453746, upper bound: 57.4573743
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2825370, upper bound: 57.4641336
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -4.9817061, 23.8439617, -32.6108971, 43.1807289
1: -11.2745037, 43.2634735, -6.3019071, 27.2622700, -38.5367699, 49.5653801
2: -11.0006275, 42.8983345, -6.4725904, 26.2338047, -37.2344322, 49.3709221
3: -19.5009098, 46.0489769, -11.1901455, 29.2783756, -48.7792854, 57.2391205
4: -17.9422951, 44.1722069, -11.0482674, 26.7482853, -44.6905785, 55.2204742

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4334759, upper bound: 57.4486622
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4334759, upper bound: 57.4702730
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.4183712, 33.1447525, -6.3430901, 28.4776020, -35.8959732, 39.4878426
1: -9.5417242, 37.5710144, -8.0922832, 32.4343948, -41.9761162, 45.6632996
2: -9.3741312, 37.0595932, -8.1187057, 31.5409679, -40.9151001, 45.1782951
3: -16.6609154, 40.0830879, -13.9889736, 34.8681641, -51.5290794, 54.0720558
4: -15.4489689, 38.1671028, -13.4151621, 32.3944283, -47.8433914, 51.5822639

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4500737, upper bound: 57.4961484
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4500737, upper bound: 57.5171266
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -7.9933391, 34.9990501, -43.8393021, 45.7563934
1: -11.2974596, 42.7681274, -10.2575874, 39.6577911, -50.9552498, 53.0257149
2: -11.0862007, 42.3602257, -10.0686836, 39.1538582, -50.2400589, 52.4289093
3: -19.3324356, 45.5596771, -17.7530479, 42.3374710, -61.6699066, 63.3127213
4: -17.8893318, 43.6359253, -16.5083942, 40.3216972, -58.2110252, 60.1443062

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5084133
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -8.4100037, 35.9523849, -44.7926331, 46.1730537
1: -11.2974596, 42.7681274, -10.7384157, 40.7441216, -52.0415802, 53.5065384
2: -11.0862007, 42.3602257, -10.5836725, 40.2473450, -51.3335457, 52.9438934
3: -19.3324356, 45.5596771, -18.3486557, 43.5231361, -62.8555717, 63.9083290
4: -17.8893318, 43.6359253, -17.1727333, 41.4359436, -59.3252678, 60.8086395

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4889722
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5102284
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.7669353, 38.1990242, -6.4958010, 29.7779331, -38.5448647, 44.6948166
1: -11.2745037, 43.2634735, -8.2680950, 33.8125229, -45.0870209, 51.5315704
2: -11.0006275, 42.8983345, -8.3059778, 33.0355911, -44.0362167, 51.2043114
3: -19.5009098, 46.0489769, -14.4991350, 36.1976891, -55.6986008, 60.5481110
4: -17.9422951, 44.1722069, -13.8112707, 33.8634415, -51.8057327, 57.9834747

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4144497, upper bound: 57.4141729
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4144497, upper bound: 57.4385264
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.4183712, 33.1447525, -7.7306695, 33.5894089, -41.0077705, 40.8754234
1: -9.5417242, 37.5710144, -9.8532286, 38.1074982, -47.6492233, 47.4242401
2: -9.3741312, 37.0595932, -9.7712288, 37.4650116, -46.8391342, 46.8308220
3: -16.6609154, 40.0830879, -16.8571243, 40.7776985, -57.4386101, 56.9402008
4: -15.4489689, 38.1671028, -15.8409624, 38.4948387, -53.9438095, 54.0080566

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4195213, upper bound: 57.4347234
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4195213, upper bound: 57.4598492
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -9.9990635, 42.8143425, -51.6545944, 47.7621117
1: -11.2974596, 42.7681274, -12.7991552, 48.4390259, -59.7364807, 55.5672836
2: -11.0862007, 42.3602257, -12.4958115, 48.1915207, -59.2777214, 54.8560371
3: -19.3324356, 45.5596771, -21.9110432, 51.5187569, -70.8511963, 67.4707108
4: -17.8893318, 43.6359253, -20.2374096, 49.5508194, -67.4401474, 63.8733330

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3518394, upper bound: 57.2438264
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3872396, upper bound: 57.2846670
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.8402529, 37.7630539, -10.0368347, 42.1911850, -51.0314369, 47.7998810
1: -11.2974596, 42.7681274, -12.7804670, 47.7420273, -59.0394783, 55.5485954
2: -11.0862007, 42.3602257, -12.5458260, 47.4610443, -58.5472450, 54.9060516
3: -19.3324356, 45.5596771, -21.6889534, 50.8226318, -70.1550674, 67.2486191
4: -17.8893318, 43.6359253, -20.1465073, 48.8336143, -66.7229385, 63.7824326

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3518394, upper bound: 57.4861891
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3872396, upper bound: 57.4965544
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -4.9817061, 23.8439617, -31.9086246, 40.2655487
1: -10.3480911, 39.9815979, -6.3019071, 27.2622700, -37.6103592, 46.2835007
2: -10.1572323, 39.4773331, -6.4725904, 26.2338047, -36.3910370, 45.9499207
3: -17.9082813, 42.6824608, -11.1901455, 29.2783756, -47.1866570, 53.8726044
4: -16.6487427, 40.6530380, -11.0482674, 26.7482853, -43.3970222, 51.7013054

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4639298
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.2892861, 32.3894920, -6.3430901, 28.4776020, -35.7668877, 38.7325783
1: -9.3475819, 36.7363091, -8.0922832, 32.4343948, -41.7819672, 44.8285904
2: -9.2219372, 36.1250496, -8.1187057, 31.5409679, -40.7629051, 44.2437515
3: -16.2676640, 39.2916412, -13.9889736, 34.8681641, -51.1358261, 53.2806129
4: -15.2287788, 37.2015991, -13.4151621, 32.3944283, -47.6232071, 50.6167603

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4659216, upper bound: 57.4976213
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4659216, upper bound: 57.5136952
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -7.9933391, 34.9990501, -43.4684296, 44.1768227
1: -10.8127460, 41.0070839, -10.2575874, 39.6577911, -50.4705353, 51.2646713
2: -10.6570034, 40.5084305, -10.0686836, 39.1538582, -49.8108597, 50.5771141
3: -18.4758568, 43.8035660, -17.7530479, 42.3374710, -60.8133278, 61.5566101
4: -17.2884102, 41.7039909, -16.5083942, 40.3216972, -57.6101074, 58.2123833

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4830663, upper bound: 57.5030042
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5248889, upper bound: 57.5166119
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -8.4100037, 35.9523849, -44.4217567, 44.5934830
1: -10.8127460, 41.0070839, -10.7384157, 40.7441216, -51.5568695, 51.7454910
2: -10.6570034, 40.5084305, -10.5836725, 40.2473450, -50.9043503, 51.0920982
3: -18.4758568, 43.8035660, -18.3486557, 43.5231361, -61.9989929, 62.1522141
4: -17.2884102, 41.7039909, -17.1727333, 41.4359436, -58.7243538, 58.8767242

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4830663, upper bound: 57.5168833
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5248889, upper bound: 57.5316336
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -9.9990635, 42.8143425, -50.8790016, 45.2829056
1: -10.3480911, 39.9815979, -12.7991552, 48.4390259, -58.7871094, 52.7807541
2: -10.1572323, 39.4773331, -12.4958115, 48.1915207, -58.3487549, 51.9731445
3: -17.9082813, 42.6824608, -21.9110432, 51.5187569, -69.4270401, 64.5934982
4: -16.6487427, 40.6530380, -20.2374096, 49.5508194, -66.1995621, 60.8904495

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3636202, upper bound: 57.2399709
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3833926, upper bound: 57.2646062
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.0646629, 35.2838440, -10.0328503, 42.1656532, -50.2303123, 45.3166924
1: -10.3480911, 39.9815979, -12.7751493, 47.7131500, -58.0612411, 52.7567482
2: -10.1572323, 39.4773331, -12.5402946, 47.4329376, -57.5901718, 52.0176201
3: -17.9082813, 42.6824608, -21.6777859, 50.7922859, -68.7005615, 64.3602448
4: -16.6487427, 40.6530380, -20.1354160, 48.8043594, -65.4531021, 60.7884521

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3636202, upper bound: 57.3778496
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3833926, upper bound: 57.3933571
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -9.9990635, 42.8143425, -51.2837181, 46.1825409
1: -10.8127460, 41.0070839, -12.7991552, 48.4390259, -59.2517700, 53.8062401
2: -10.6570034, 40.5084305, -12.4958115, 48.1915207, -58.8485260, 53.0042419
3: -18.4758568, 43.8035660, -21.9110432, 51.5187569, -69.9946136, 65.7145920
4: -17.2884102, 41.7039909, -20.2374096, 49.5508194, -66.8392334, 61.9413986

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3887402, upper bound: 57.2659148
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4144551, upper bound: 57.2944179
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.4693804, 36.1834831, -10.0368347, 42.1911850, -50.6605606, 46.2203140
1: -10.8127460, 41.0070839, -12.7804670, 47.7420273, -58.5547676, 53.7875519
2: -10.6570034, 40.5084305, -12.5458260, 47.4610443, -58.1180496, 53.0542564
3: -18.4758568, 43.8035660, -21.6889534, 50.8226318, -69.2984924, 65.4925079
4: -17.2884102, 41.7039909, -20.1465073, 48.8336143, -66.1220245, 61.8504982

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3593944, upper bound: 57.5367369
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4144551, upper bound: 57.5483706
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -8.7669353, 38.1990242, -48.8947754, 53.7135124
1: -13.6549959, 50.8562546, -11.2745037, 43.2634735, -56.9184685, 62.1307526
2: -13.3063459, 50.7025528, -11.0006275, 42.8983345, -56.2046814, 61.7031593
3: -23.1477661, 54.0001144, -19.5009098, 46.0489769, -69.1967468, 73.5010223
4: -21.2595654, 52.1365585, -17.9422951, 44.1722069, -65.4317703, 70.0788498

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.4793470, upper bound: 56.6158258
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4002566, upper bound: 57.4451127
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -8.8402529, 37.7630539, -48.4588013, 53.7868347
1: -13.6549959, 50.8562546, -11.2974596, 42.7681274, -56.4231224, 62.1537132
2: -13.3063459, 50.7025528, -11.0862007, 42.3602257, -55.6665726, 61.7887306
3: -23.1477661, 54.0001144, -19.3324356, 45.5596771, -68.7074432, 73.3325500
4: -21.2595654, 52.1365585, -17.8893318, 43.6359253, -64.8954926, 70.0258713

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.4793470, upper bound: 56.9036542
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4002566, upper bound: 57.4791871
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.0847101, 47.3818054, -8.4693804, 36.1834831, -47.2681885, 55.8511772
1: -14.2560978, 53.6242714, -10.8127460, 41.0070839, -55.2631798, 64.4370193
2: -13.7948971, 53.5095329, -10.6570034, 40.5084305, -54.3033218, 64.1665344
3: -24.3539143, 56.8776131, -18.4758568, 43.8035660, -68.1574783, 75.3534698
4: -22.2534161, 54.9756432, -17.2884102, 41.7039909, -63.9574051, 72.2640533

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -8.0646629, 35.2838440, -45.9795990, 53.0112419
1: -13.6549959, 50.8562546, -10.3480911, 39.9815979, -53.6365929, 61.2043419
2: -13.3063459, 50.7025528, -10.1572323, 39.4773331, -52.7836800, 60.8597641
3: -23.1477661, 54.0001144, -17.9082813, 42.6824608, -65.8302307, 71.9083939
4: -21.2595654, 52.1365585, -16.6487427, 40.6530380, -61.9126053, 68.7853012

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2510664, upper bound: 57.3882499
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1963502, upper bound: 57.2887893
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2794578, upper bound: 57.4050682
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -8.4693804, 36.1834831, -46.8792305, 53.4159622
1: -13.6549959, 50.8562546, -10.8127460, 41.0070839, -54.6620789, 61.6689987
2: -13.3063459, 50.7025528, -10.6570034, 40.5084305, -53.8147736, 61.3595505
3: -23.1477661, 54.0001144, -18.4758568, 43.8035660, -66.9513321, 72.4759674
4: -21.2595654, 52.1365585, -17.2884102, 41.7039909, -62.9635544, 69.4249725

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2510664, upper bound: 57.4585025
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1963502, upper bound: 57.4724474
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2794578, upper bound: 57.4699324
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -8.7486715, 38.1090698, -48.1284485, 51.6430511
1: -12.8247375, 48.5298729, -11.2512512, 43.1552277, -55.9799576, 59.7811203
2: -12.5208769, 48.2823448, -10.9772034, 42.7888718, -55.3097420, 59.2595444
3: -21.9548664, 51.6154327, -19.4641361, 45.9340324, -67.8888855, 71.0795670
4: -20.2769165, 49.6436996, -17.8985062, 44.0687370, -64.3456573, 67.5422058

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2420246, upper bound: 57.3615723
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -8.8337555, 37.7311478, -47.7505264, 51.7281380
1: -12.8247375, 48.5298729, -11.2892466, 42.7298050, -55.5545387, 59.8191185
2: -12.5208769, 48.2823448, -11.0780182, 42.3217621, -54.8426361, 59.3603554
3: -21.9548664, 51.6154327, -19.3194389, 45.5189896, -67.4738541, 70.9348755
4: -20.2769165, 49.6436996, -17.8740311, 43.5991898, -63.8761063, 67.5177307

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2420246, upper bound: 57.3615723
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -8.7669353, 38.1990242, -48.2362556, 50.9596329
1: -12.7809610, 47.7437553, -11.2745037, 43.2634735, -56.0444336, 59.0182419
2: -12.5463152, 47.4627686, -11.0006275, 42.8983345, -55.4446487, 58.4633942
3: -21.6898041, 50.8244820, -19.5009098, 46.0489769, -67.7387772, 70.3253937
4: -20.1472874, 48.8353882, -17.9422951, 44.1722069, -64.3194885, 66.7776794

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4385264, upper bound: 57.4741026
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4598491, upper bound: 57.4794884
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -8.8402529, 37.7630539, -47.8002815, 51.0329590
1: -12.7809610, 47.7437553, -11.2974596, 42.7681274, -55.5490875, 59.0412064
2: -12.5463152, 47.4627686, -11.0862007, 42.3602257, -54.9065399, 58.5489693
3: -21.6898041, 50.8244820, -19.3324356, 45.5596771, -67.2494736, 70.1569214
4: -20.1472874, 48.8353882, -17.8893318, 43.6359253, -63.7831993, 66.7246933

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4385264, upper bound: 57.4771422
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4598491, upper bound: 57.4897751
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -8.0646629, 35.2838440, -45.3032227, 50.9590416
1: -12.8247375, 48.5298729, -10.3480911, 39.9815979, -52.8063354, 58.8779640
2: -12.5208769, 48.2823448, -10.1572323, 39.4773331, -51.9982033, 58.4395714
3: -21.9548664, 51.6154327, -17.9082813, 42.6824608, -64.6373291, 69.5237122
4: -20.2769165, 49.6436996, -16.6487427, 40.6530380, -60.9299545, 66.2924423

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2488680, upper bound: 57.3784279
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4051504
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -8.4693804, 36.1834831, -46.2028618, 51.3637657
1: -12.8247375, 48.5298729, -10.8127460, 41.0070839, -53.8318214, 59.3426208
2: -12.5208769, 48.2823448, -10.6570034, 40.5084305, -53.0292969, 58.9393463
3: -21.9548664, 51.6154327, -18.4758568, 43.8035660, -65.7584305, 70.0912933
4: -20.2769165, 49.6436996, -17.2884102, 41.7039909, -61.9809074, 66.9321136

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2488680, upper bound: 57.3816615
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4081359
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -8.0646629, 35.2838440, -45.3210754, 50.2573624
1: -12.7809610, 47.7437553, -10.3480911, 39.9815979, -52.7625580, 58.0918427
2: -12.5463152, 47.4627686, -10.1572323, 39.4773331, -52.0236473, 57.6200027
3: -21.6898041, 50.8244820, -17.9082813, 42.6824608, -64.3722687, 68.7327652
4: -20.1472874, 48.8353882, -16.6487427, 40.6530380, -60.8003235, 65.4841232

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4729352, upper bound: 57.4989517
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5130108, upper bound: 57.5115561
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -8.4693804, 36.1834831, -46.2207108, 50.6620827
1: -12.7809610, 47.7437553, -10.8127460, 41.0070839, -53.7880440, 58.5564957
2: -12.5463152, 47.4627686, -10.6570034, 40.5084305, -53.0547447, 58.1197739
3: -21.6898041, 50.8244820, -18.4758568, 43.8035660, -65.4933548, 69.3003387
4: -20.1472874, 48.8353882, -17.2884102, 41.7039909, -61.8512764, 66.1237946

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4729352, upper bound: 57.5249359
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5130108, upper bound: 57.5335463
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -10.6957531, 44.9465866, -55.6423340, 55.6423340
1: -13.6549959, 50.8562546, -13.6549959, 50.8562546, -64.5112381, 64.5112381
2: -13.3063459, 50.7025528, -13.3063459, 50.7025528, -64.0088882, 64.0088882
3: -23.1477661, 54.0001144, -23.1477661, 54.0001144, -77.1478806, 77.1478806
4: -21.2595654, 52.1365585, -21.2595654, 52.1365585, -73.3961258, 73.3961258

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.3771656, upper bound: 56.8752642
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1825066, upper bound: 57.4663902
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.6957531, 44.9465866, -10.0372343, 42.1927071, -52.8884544, 54.9838104
1: -13.6549959, 50.8562546, -12.7809610, 47.7437553, -61.3987465, 63.6372147
2: -13.3063459, 50.7025528, -12.5463152, 47.4627686, -60.7691154, 63.2488441
3: -23.1477661, 54.0001144, -21.6898041, 50.8244820, -73.9722443, 75.6899185
4: -21.2595654, 52.1365585, -20.1472874, 48.8353882, -70.0949554, 72.2838364

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.2514034, upper bound: 56.7878147
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1728857, upper bound: 57.4856520
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -10.6957531, 44.9465866, -54.9838104, 52.8884544
1: -12.7809610, 47.7437553, -13.6549959, 50.8562546, -63.6372070, 61.3987427
2: -12.5463152, 47.4627686, -13.3063459, 50.7025528, -63.2488441, 60.7691154
3: -21.6898041, 50.8244820, -23.1477661, 54.0001144, -75.6899185, 73.9722443
4: -20.1472874, 48.8353882, -21.2595654, 52.1365585, -72.2838440, 70.0949554

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.1748844, upper bound: 56.5384255
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2719423, upper bound: 57.4791578
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.0193787, 42.8943863, -10.0328503, 42.1656532, -52.1850319, 52.9272346
1: -12.8247375, 48.5298729, -12.7751493, 47.7131500, -60.5378876, 61.3050232
2: -12.5208769, 48.2823448, -12.5402946, 47.4329376, -59.9538155, 60.8226395
3: -21.9548664, 51.6154327, -21.6777859, 50.7922859, -72.7471466, 73.2932205
4: -20.2769165, 49.6436996, -20.1354160, 48.8043594, -69.0812759, 69.7791138

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -55.9521207, upper bound: 55.9099219
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2028543, upper bound: 57.3475413
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -10.0193787, 42.8943863, -52.9316216, 52.2120857
1: -12.7809610, 47.7437553, -12.8247375, 48.5298729, -61.3108330, 60.5684929
2: -12.5463152, 47.4627686, -12.5208769, 48.2823448, -60.8286591, 59.9836464
3: -21.6898041, 50.8244820, -21.9548664, 51.6154327, -73.3052368, 72.7793503
4: -20.1472874, 48.8353882, -20.2769165, 49.6436996, -69.7909851, 69.1122971

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.1409986, upper bound: 56.1932261
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4151434, upper bound: 57.2946578
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.0372343, 42.1927071, -10.0372343, 42.1927071, -52.2299347, 52.2299347
1: -12.7809610, 47.7437553, -12.7809610, 47.7437553, -60.5247002, 60.5247154
2: -12.5463152, 47.4627686, -12.5463152, 47.4627686, -60.0090752, 60.0090828
3: -21.6898041, 50.8244820, -21.6898041, 50.8244820, -72.5142822, 72.5142822
4: -20.1472874, 48.8353882, -20.1472874, 48.8353882, -68.9826584, 68.9826584

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.1409986, upper bound: 56.5384255
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4151434, upper bound: 57.5619912
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.97 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4200260
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4402933
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4986064
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3474123, upper bound: 57.3359198
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4227593, upper bound: 57.3625036
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2172797, upper bound: 57.4566298
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2172797, upper bound: 57.4641336
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4250266
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4417602
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4604355, upper bound: 57.5000732
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4702730, upper bound: 57.4865799
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5167745, upper bound: 57.5017011
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4702730, upper bound: 57.4865799
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5167745, upper bound: 57.5017011
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2443625, upper bound: 57.1149556
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2804985, upper bound: 57.2922389
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2453746, upper bound: 57.0864061
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2825370, upper bound: 57.1263918
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2453746, upper bound: 57.4573743
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2825370, upper bound: 57.4641336
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4334759, upper bound: 57.4486622
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4334759, upper bound: 57.4702730
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4500737, upper bound: 57.4961484
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4500737, upper bound: 57.5171266
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5084133
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4889722
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5102284
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4144497, upper bound: 57.4141729
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4144497, upper bound: 57.4385264
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4195213, upper bound: 57.4347234
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4195213, upper bound: 57.4598492
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3518394, upper bound: 57.2438264
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3872396, upper bound: 57.2846670
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3518394, upper bound: 57.4861891
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3872396, upper bound: 57.4965544
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4639298
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4659216, upper bound: 57.4976213
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4659216, upper bound: 57.5136952
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4830663, upper bound: 57.5030042
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5248889, upper bound: 57.5166119
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4830663, upper bound: 57.5168833
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5248889, upper bound: 57.5316336
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3636202, upper bound: 57.2399709
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3833926, upper bound: 57.2646062
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3636202, upper bound: 57.3778496
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3833926, upper bound: 57.3933571
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3887402, upper bound: 57.2659148
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4144551, upper bound: 57.2944179
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.3593944, upper bound: 57.5367369
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4144551, upper bound: 57.5483706
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.4793470, upper bound: 56.6158258
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4002566, upper bound: 57.4451127
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.4793470, upper bound: 56.9036542
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4002566, upper bound: 57.4791871
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.1963502, upper bound: 57.2887893
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2794578, upper bound: 57.4050682
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.1963502, upper bound: 57.4724474
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2794578, upper bound: 57.4699324
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2420246, upper bound: 57.3615723
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2420246, upper bound: 57.3615723
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2869449, upper bound: 57.3892579
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4385264, upper bound: 57.4741026
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4598491, upper bound: 57.4794884
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4385264, upper bound: 57.4771422
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4598491, upper bound: 57.4897751
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2488680, upper bound: 57.3784279
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4051504
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2488680, upper bound: 57.3816615
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2933168, upper bound: 57.4081359
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4729352, upper bound: 57.4989517
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5130108, upper bound: 57.5115561
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4729352, upper bound: 57.5249359
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.5130108, upper bound: 57.5335463
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.3771656, upper bound: 56.8752642
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.1825066, upper bound: 57.4663902
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.2514034, upper bound: 56.7878147
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.1728857, upper bound: 57.4856520
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.1748844, upper bound: 56.5384255
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2719423, upper bound: 57.4791578
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -55.9521207, upper bound: 55.9099219
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.2028543, upper bound: 57.3475413
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.1409986, upper bound: 56.1932261
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4151434, upper bound: 57.2946578
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -56.1409986, upper bound: 56.5384255
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -57.4151434, upper bound: 57.5619912

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.7451043, 27.7374706, -5.4138098, 25.6408691, -31.3859711, 33.1512794
1: -7.3574800, 31.5693779, -6.8880334, 29.2226677, -36.5801468, 38.4574089
2: -7.3863897, 30.7097282, -6.9775701, 28.3149376, -35.7013283, 37.6872978
3: -13.2385550, 33.7216721, -12.2375631, 31.2964668, -44.5350227, 45.9592361
4: -12.5734367, 31.4742508, -11.7668877, 28.9843082, -41.5577469, 43.2411385

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4200260
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4200260
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.6024351, 30.2522888, -5.4138098, 25.6408691, -32.2433052, 35.6660995
1: -8.4603643, 34.3885117, -6.8880334, 29.2226677, -37.6830330, 41.2765465
2: -8.4155941, 33.6497993, -6.9775701, 28.3149376, -36.7305260, 40.6273689
3: -14.8398304, 36.7894897, -12.2375631, 31.2964668, -46.1362953, 49.0270538
4: -13.9825315, 34.5861893, -11.7668877, 28.9843082, -42.9668388, 46.3530769

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4402933
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4402933
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.7451043, 27.7374706, -6.4946232, 29.0429840, -34.7880859, 34.2320938
1: -7.3574800, 31.5693779, -8.3029099, 33.0288963, -40.3863754, 39.8722878
2: -7.3863897, 30.7097282, -8.2685080, 32.2582207, -39.6446075, 38.9782372
3: -13.2385550, 33.7216721, -14.3753710, 35.4210663, -48.6596222, 48.0970421
4: -12.5734367, 31.4742508, -13.5885582, 33.1541367, -45.7275734, 45.0628090

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3555274, upper bound: 57.4376124
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4214010, upper bound: 57.4751627
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.6024351, 30.2522888, -6.4946232, 29.0429840, -35.6454201, 36.7469101
1: -8.4603643, 34.3885117, -8.3029099, 33.0288963, -41.4892616, 42.6914215
2: -8.4155941, 33.6497993, -8.2685080, 32.2582207, -40.6738129, 41.9183083
3: -14.8398304, 36.7894897, -14.3753710, 35.4210663, -50.2608948, 51.1648560
4: -13.9825315, 34.5861893, -13.5885582, 33.1541367, -47.1366653, 48.1747398

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3555274, upper bound: 57.4610799
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4214010, upper bound: 57.4954863
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9486384, 27.6570396, -8.9847889, 38.9609985, -44.9096375, 36.6418304
1: -7.5847530, 31.4490318, -11.5671473, 44.1251221, -51.7098770, 43.0161743
2: -7.6243067, 30.6088829, -11.2372074, 43.7923737, -51.4166794, 41.8460846
3: -13.3943548, 33.6805496, -19.9395885, 46.9269409, -60.3212891, 53.6201363
4: -12.7465086, 31.4047012, -18.2945824, 45.0634575, -57.8099670, 49.6992836

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4200260, upper bound: 57.4235198
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4200260, upper bound: 57.4801864
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0722103, 31.2705612, -7.6448956, 33.9697990, -41.0420074, 38.9154472
1: -9.0512457, 35.5169983, -9.8463955, 38.5017319, -47.5529785, 45.3633957
2: -8.9803925, 34.8309250, -9.6262989, 38.0226326, -47.0030251, 44.4572220
3: -15.6104374, 38.0437126, -17.1318951, 41.0299911, -56.6404266, 55.1756058
4: -14.6354523, 35.8261719, -15.8260746, 39.1356659, -53.7711182, 51.6522446

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4783391, upper bound: 57.4444451
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4783391, upper bound: 57.5011117
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9486384, 27.6570396, -9.0520763, 38.5367279, -44.4853668, 36.7091141
1: -7.5847530, 31.4490318, -11.5747042, 43.6418037, -51.2265549, 43.0237312
2: -7.6243067, 30.6088829, -11.3216534, 43.2610741, -50.8853798, 41.9305305
3: -13.3943548, 33.6805496, -19.7602921, 46.4578094, -59.8521461, 53.4408379
4: -12.7465086, 31.4047012, -18.2454472, 44.5368423, -57.2833405, 49.6501465

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4155529, upper bound: 57.4155529
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4155529, upper bound: 57.4738660
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.0722103, 31.2705612, -7.8063006, 33.8532486, -40.9254608, 39.0768547
1: -9.0512457, 35.5169983, -9.9959126, 38.3750801, -47.4263268, 45.5129089
2: -8.9803925, 34.8309250, -9.8215570, 37.8676643, -46.8480530, 44.6524811
3: -15.6104374, 38.0437126, -17.1718159, 40.9313049, -56.5417404, 55.2155266
4: -14.6354523, 35.8261719, -15.9295483, 38.9899254, -53.6253777, 51.7557182

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4738660, upper bound: 57.4364782
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4738660, upper bound: 57.4947912
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.7451043, 27.7374706, -9.7917204, 41.4309692, -47.1760674, 37.5291901
1: -7.3574800, 31.5693779, -12.5148449, 46.8862114, -54.2436905, 44.0842209
2: -7.3863897, 30.7097282, -12.2084789, 46.6534081, -54.0397949, 42.9182053
3: -13.2385550, 33.7216721, -21.2993679, 49.8507919, -63.0893478, 55.0210419
4: -12.5734367, 31.4742508, -19.5878906, 48.0006866, -60.5741234, 51.0621414

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3474123, upper bound: 57.3359198
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3474123, upper bound: 57.3359198
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.6024351, 30.2522888, -8.5017281, 36.5821915, -43.1846237, 38.7540169
1: -8.4603643, 34.3885117, -10.8708820, 41.4128799, -49.8732452, 45.2593918
2: -8.4155941, 33.6497993, -10.6533527, 41.0644226, -49.4800110, 44.3031540
3: -14.8398304, 36.7894897, -18.5997925, 44.0996590, -58.9394913, 55.3892822
4: -13.9825315, 34.5861893, -17.1714382, 42.2550278, -56.2375603, 51.7576294

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4227594, upper bound: 57.3625036
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4227594, upper bound: 57.3625036
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.5132961, 32.8513870, -10.3668098, 43.6890259, -51.2023163, 43.2181969
1: -9.6256981, 37.2470627, -13.2427340, 49.4362907, -59.0619736, 50.4897957
2: -9.4870529, 36.6835403, -12.9066353, 49.2585182, -58.7455711, 49.5901756
3: -16.5858116, 39.7460976, -22.4675522, 52.5081940, -69.0940094, 62.2136421
4: -15.4189711, 37.7691879, -20.6345272, 50.6546936, -66.0736618, 58.4037132

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4032350, upper bound: 57.3826352
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4070785, upper bound: 57.3776411
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.2863541, 35.6743469, -10.6957531, 44.9465866, -53.2329369, 46.3701019
1: -10.6042032, 40.4184189, -13.6549959, 50.8562546, -61.4604530, 54.0734100
2: -10.4154091, 39.9502335, -13.3063459, 50.7025528, -61.1179466, 53.2565765
3: -18.1936722, 43.0959473, -23.1477661, 54.0001144, -72.1937790, 66.2436981
4: -16.8570976, 41.1582947, -21.2595654, 52.1365585, -68.9936523, 62.4178619

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494994, upper bound: 57.4471125
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4495017, upper bound: 57.4338265
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.2912812, 25.6779537, -5.4138098, 25.6408691, -30.9321499, 31.0917625
1: -6.7747488, 29.3011360, -6.8880334, 29.2226677, -35.9974136, 36.1891670
2: -6.8423648, 28.3550415, -6.9775701, 28.3149376, -35.1573029, 35.3326111
3: -12.1550131, 31.3615494, -12.2375631, 31.2964668, -43.4514694, 43.5991096
4: -11.7054529, 29.0024853, -11.7668877, 28.9843082, -40.6897621, 40.7693710

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4250266
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4250266
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.7937622, 30.7798309, -5.4138098, 25.6408691, -32.4346313, 36.1936417
1: -8.6969538, 35.0070839, -6.8880334, 29.2226677, -37.9196167, 41.8951187
2: -8.6619501, 34.1906509, -6.9775701, 28.3149376, -36.9768867, 41.1682205
3: -15.1779556, 37.5357819, -12.2375631, 31.2964668, -46.4744225, 49.7733421
4: -14.3899021, 35.1436996, -11.7668877, 28.9843082, -43.3742104, 46.9105873

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4417602
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4417602
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.2425332, 25.4666214, -6.4946232, 29.0429840, -34.2855148, 31.9612446
1: -6.7157836, 29.0644989, -8.3029099, 33.0288963, -39.7446747, 37.3674088
2: -6.7829127, 28.1207294, -8.2685080, 32.2582207, -39.0411301, 36.3892365
3: -12.0501604, 31.1102924, -14.3753710, 35.4210663, -47.4712257, 45.4856606
4: -11.6059141, 28.7633476, -13.5885582, 33.1541367, -44.7600517, 42.3518944

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3224120, upper bound: 57.4376124
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4345702, upper bound: 57.4794759
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.7937622, 30.7798309, -6.4946232, 29.0429840, -35.8367462, 37.2744522
1: -8.6969538, 35.0070839, -8.3029099, 33.0288963, -41.7258453, 43.3099937
2: -8.6619501, 34.1906509, -8.2685080, 32.2582207, -40.9201698, 42.4591599
3: -15.1779556, 37.5357819, -14.3753710, 35.4210663, -50.5990181, 51.9111443
4: -14.3899021, 35.1436996, -13.5885582, 33.1541367, -47.5440369, 48.7322464

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3224120, upper bound: 57.4593290
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4345702, upper bound: 57.4960641
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.5963645, 26.3081665, -8.9847889, 38.9609985, -44.5573616, 35.2929535
1: -7.1001191, 29.9992542, -11.5671473, 44.1251221, -51.2252426, 41.5663986
2: -7.2258878, 29.0324535, -11.2372074, 43.7923737, -51.0182610, 40.2696571
3: -12.5410995, 32.1875839, -19.9395885, 46.9269409, -59.4680367, 52.1271706
4: -12.1948471, 29.6890259, -18.2945824, 45.0634575, -57.2583046, 47.9836082

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4486622, upper bound: 57.4334759
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4486622, upper bound: 57.4906093
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.9877100, 30.9087677, -7.6448956, 33.9697990, -40.9575081, 38.5536575
1: -8.9331646, 35.1532211, -9.8463955, 38.5017319, -47.4348984, 44.9996185
2: -8.8986206, 34.3513451, -9.6262989, 38.0226326, -46.9212532, 43.9776459
3: -15.3714523, 37.7349548, -17.1318951, 41.0299911, -56.4014435, 54.8668518
4: -14.5742207, 35.3111000, -15.8260746, 39.1356659, -53.7098846, 51.1371765

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4961484, upper bound: 57.4500737
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4961484, upper bound: 57.5069508
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.5963645, 26.3081665, -9.0520763, 38.5367279, -44.1330872, 35.3602409
1: -7.1001191, 29.9992542, -11.5747042, 43.6418037, -50.7419243, 41.5739555
2: -7.2258878, 29.0324535, -11.3216534, 43.2610741, -50.4869614, 40.3541069
3: -12.5410995, 32.1875839, -19.7602921, 46.4578094, -58.9988937, 51.9478722
4: -12.1948471, 29.6890259, -18.2454472, 44.5368423, -56.7316895, 47.9344711

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4497101, upper bound: 57.4280291
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4497101, upper bound: 57.4865799
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.9877100, 30.9087677, -7.8063006, 33.8532486, -40.8409576, 38.7150612
1: -8.9331646, 35.1532211, -9.9959126, 38.3750801, -47.3082428, 45.1491280
2: -8.8986206, 34.3513451, -9.8215570, 37.8676643, -46.7662849, 44.1729012
3: -15.3714523, 37.7349548, -17.1718159, 40.9313049, -56.3027573, 54.9067688
4: -14.5742207, 35.3111000, -15.9295483, 38.9899254, -53.5641479, 51.2406349

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4955819, upper bound: 57.4432771
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4738660, upper bound: 57.5017011
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.4526935, 32.9576874, -10.4994717, 44.1784554, -51.6311417, 43.4571571
1: -9.5677376, 37.3750801, -13.4083557, 49.9917030, -59.5594330, 50.7834320
2: -9.4119253, 36.7907982, -13.0679817, 49.8232117, -59.2351303, 49.8587761
3: -16.6255531, 39.9388885, -22.7416935, 53.0925331, -69.7180710, 62.6805801
4: -15.4883995, 37.9122391, -20.8840942, 51.2441177, -66.7325058, 58.7963333

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2887893, upper bound: 57.1963502
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4050682, upper bound: 57.2794578
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.8146300, 33.8448753, -11.0243721, 47.1488724, -54.9635010, 44.8692474
1: -9.9862337, 38.3880119, -14.1803865, 53.3598557, -63.3460846, 52.5683975
2: -9.8671780, 37.7987099, -13.7203350, 53.2445908, -63.1117554, 51.5190430
3: -17.1413631, 41.0422592, -24.2248249, 56.5950623, -73.7363892, 65.2670822
4: -16.0861244, 38.9029541, -22.1343040, 54.7041283, -70.7902527, 61.0372581

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3397822, 32.0293846, -10.3668098, 43.6890259, -51.0288048, 42.3961945
1: -9.3865557, 36.3528557, -13.2427340, 49.4362907, -58.8228455, 49.5955887
2: -9.2975655, 35.7077560, -12.9066353, 49.2585182, -58.5560837, 48.6143913
3: -16.1304836, 38.8894463, -22.4675522, 52.5081940, -68.6386719, 61.3569870
4: -15.1723795, 36.7471008, -20.6345272, 50.6546936, -65.8270721, 57.3816223

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3908385, upper bound: 57.2902188
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4024347, upper bound: 57.3169200
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.8146300, 33.8448753, -10.6957531, 44.9465866, -52.7612152, 44.5406265
1: -9.9862337, 38.3880119, -13.6549959, 50.8562546, -60.8424873, 52.0430031
2: -9.8671780, 37.7987099, -13.3063459, 50.7025528, -60.5697212, 51.1050568
3: -17.1413631, 41.0422592, -23.1477661, 54.0001144, -71.1414642, 64.1900253
4: -16.0861244, 38.9029541, -21.2595654, 52.1365585, -68.2226868, 60.1625214

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494994, upper bound: 57.4337955
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4595346, upper bound: 57.4338041
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.7451043, 27.7374706, -4.9817061, 23.8439617, -29.5890656, 32.7191772
1: -7.3574800, 31.5693779, -6.3019071, 27.2622700, -34.6197510, 37.8712769
2: -7.3863897, 30.7097282, -6.4725904, 26.2338047, -33.6201935, 37.1823158
3: -13.2385550, 33.7216721, -11.1901455, 29.2783756, -42.5169296, 44.9118195
4: -12.5734367, 31.4742508, -11.0482674, 26.7482853, -39.3217239, 42.5225182

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4439833
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4486622
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.6024351, 30.2522888, -4.9817061, 23.8439617, -30.4463959, 35.2339935
1: -8.4603643, 34.3885117, -6.3019071, 27.2622700, -35.7226334, 40.6904144
2: -8.4155941, 33.6497993, -6.4725904, 26.2338047, -34.6493912, 40.1223869
3: -14.8398304, 36.7894897, -11.1901455, 29.2783756, -44.1182060, 47.9796371
4: -13.9825315, 34.5861893, -11.0482674, 26.7482853, -40.7308083, 45.6344528

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4642506
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4702730
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.7451043, 27.7374706, -6.3430901, 28.4776020, -34.2227020, 34.0805588
1: -7.3574800, 31.5693779, -8.0922832, 32.4343948, -39.7918739, 39.6616592
2: -7.3863897, 30.7097282, -8.1187057, 31.5409679, -38.9273567, 38.8284264
3: -13.2385550, 33.7216721, -13.9889736, 34.8681641, -48.1067200, 47.7106476
4: -12.5734367, 31.4742508, -13.4151621, 32.3944283, -44.9678650, 44.8894119

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4919611
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4961483
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.6024351, 30.2522888, -6.3430901, 28.4776020, -35.0800362, 36.5953789
1: -8.4603643, 34.3885117, -8.0922832, 32.4343948, -40.8947563, 42.4807968
2: -8.4155941, 33.6497993, -8.1187057, 31.5409679, -39.9565582, 41.7684975
3: -14.8398304, 36.7894897, -13.9889736, 34.8681641, -49.7079926, 50.7784615
4: -13.9825315, 34.5861893, -13.4151621, 32.3944283, -46.3769531, 48.0013504

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.5122283
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.5122283
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9486384, 27.6570396, -7.9933391, 34.9990501, -40.9476891, 35.6503792
1: -7.5847530, 31.4490318, -10.2575874, 39.6577911, -47.2425461, 41.7066193
2: -7.6243067, 30.6088829, -10.0686836, 39.1538582, -46.7781639, 40.6775665
3: -13.3943548, 33.6805496, -17.7530479, 42.3374710, -55.7318230, 51.4335976
4: -12.7465086, 31.4047012, -16.5083942, 40.3216972, -53.0681992, 47.9130936

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4395102
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4874880
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0722103, 31.2705612, -7.2310572, 32.1572723, -39.2294807, 38.5016174
1: -9.0512457, 35.5169983, -9.2735844, 36.4722443, -45.5234909, 44.7905807
2: -8.9803925, 34.8309250, -9.1495342, 35.8610153, -44.8414078, 43.9804611
3: -15.6104374, 38.0437126, -16.1408138, 39.0104828, -54.6209183, 54.1845245
4: -14.6354523, 35.8261719, -15.1141539, 36.9309273, -51.5663795, 50.9403267

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4833397, upper bound: 57.4604355
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4833397, upper bound: 57.5084132
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9486384, 27.6570396, -8.4100037, 35.9523849, -41.9010201, 36.0670433
1: -7.5847530, 31.4490318, -10.7384157, 40.7441216, -48.3288727, 42.1874390
2: -7.6243067, 30.6088829, -10.5836725, 40.2473450, -47.8716507, 41.1925545
3: -13.3943548, 33.6805496, -18.3486557, 43.5231361, -56.9174881, 52.0292053
4: -12.7465086, 31.4047012, -17.1727333, 41.4359436, -54.1824455, 48.5774345

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4434373
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4889722
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.0722103, 31.2705612, -7.3956771, 32.1658363, -39.2380447, 38.6662331
1: -9.0512457, 35.5169983, -9.4487677, 36.5041809, -45.5554276, 44.9657631
2: -8.9803925, 34.8309250, -9.3614864, 35.8843689, -44.8647614, 44.1924133
3: -15.6104374, 38.0437126, -16.2342567, 39.0703354, -54.6807709, 54.2779655
4: -14.6354523, 35.8261719, -15.2833891, 36.9460907, -51.5815430, 51.1095619

Time for backsubstitution: 1.77 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.12 + 418.47 = 421.59 seconds
