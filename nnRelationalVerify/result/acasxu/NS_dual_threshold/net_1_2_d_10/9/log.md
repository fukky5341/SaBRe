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
execution time: IAR + RelationalAnalysis = 1.66 + 1.58 = 3.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -57.5687468, upper bound: 57.5687468

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5532159, upper bound: 57.5570901
time: 0.47 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5621985, upper bound: 57.5621985
time: 0.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.12 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 0, lower bound: -57.5532159, upper bound: 57.5570901
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 0, lower bound: -57.5621985, upper bound: 57.5621985

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.1770201, 42.4380188, -11.8764477, 48.9575691, -59.1345901, 54.3144684
1: -12.9492474, 48.0443954, -15.0760498, 55.3899689, -68.3392181, 63.1204453
2: -12.7204800, 47.7395439, -14.7838812, 55.2434998, -67.9639816, 62.5234261
3: -21.9290562, 51.2056084, -25.3967628, 58.8714714, -80.8005219, 76.6023712
4: -20.4464874, 49.1660233, -23.6019707, 56.9266968, -77.3731613, 72.7679749

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5570901
time: 0.46 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -13.0451412, 53.5286636, -65.5080032, 62.5160027
1: -15.2106190, 55.9559326, -16.5472050, 60.5470352, -75.7576523, 72.5031357
2: -14.9106045, 55.8369446, -16.2069473, 60.5032959, -75.4138870, 72.0438919
3: -25.6317139, 59.4555740, -27.8260193, 64.2862701, -89.9179688, 87.2815933
4: -23.7928352, 57.5261345, -25.8074226, 62.3490105, -86.1418457, 83.3335419

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
time: 0.45 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621985
time: 0.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.58 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5570901
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621985

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -10.1770201, 42.4380188, -10.1770201, 42.4380188, -52.6150398, 52.6150398
1: -12.9492474, 48.0443954, -12.9492474, 48.0443954, -60.9936447, 60.9936447
2: -12.7204800, 47.7395439, -12.7204800, 47.7395439, -60.4600182, 60.4600143
3: -21.9290562, 51.2056084, -21.9290562, 51.2056084, -73.1346588, 73.1346588
4: -20.4464874, 49.1660233, -20.4464874, 49.1660233, -69.6124954, 69.6124954

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5210888
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.48 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.1770201, 42.4380188, -11.9793425, 49.4708633, -59.6478844, 54.4173622
1: -12.9492474, 48.0443954, -15.2106190, 55.9559326, -68.9051743, 63.2550125
2: -12.7204800, 47.7395439, -14.9106045, 55.8369446, -68.5574265, 62.6501427
3: -21.9290562, 51.2056084, -25.6317139, 59.4555740, -81.3846283, 76.8373108
4: -20.4464874, 49.1660233, -23.7928352, 57.5261345, -77.9726028, 72.9588547

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5210888
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5570901
time: 0.49 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -10.1770201, 42.4380188, -54.4173622, 59.6478844
1: -15.2106190, 55.9559326, -12.9492474, 48.0443954, -63.2550125, 68.9051743
2: -14.9106045, 55.8369446, -12.7204800, 47.7395439, -62.6501465, 68.5574265
3: -25.6317139, 59.4555740, -21.9290562, 51.2056084, -76.8373108, 81.3846283
4: -23.7928352, 57.5261345, -20.4464874, 49.1660233, -72.9588547, 77.9725952

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4959195, upper bound: 57.4797424
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
time: 0.52 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -11.9793425, 49.4708633, -61.4502068, 61.4502068
1: -15.2106190, 55.9559326, -15.2106190, 55.9559326, -71.1665497, 71.1665497
2: -14.9106045, 55.8369446, -14.9106045, 55.8369446, -70.7475510, 70.7475510
3: -25.6317139, 59.4555740, -25.6317139, 59.4555740, -85.0872879, 85.0872879
4: -23.7928352, 57.5261345, -23.7928352, 57.5261345, -81.3189545, 81.3189621

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4959195, upper bound: 57.4797424
time: 0.52 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621587
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.74 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5210888
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.5063429, upper bound: 57.5210888
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5570901
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.4959195, upper bound: 57.4797424
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5532159
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.4959195, upper bound: 57.4797424
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -57.5570901, upper bound: 57.5621587

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -9.6038160, 40.2809982, -49.4385834, 48.4782448
1: -11.6959438, 44.0212021, -12.2323484, 45.6113548, -57.3072968, 56.2535439
2: -11.4683819, 43.6604576, -12.0197229, 45.2531128, -56.7214928, 55.6801720
3: -19.9659348, 46.8748016, -20.7653065, 48.6373138, -68.6032333, 67.6400909
4: -18.4506874, 44.9865723, -19.3651485, 46.5791321, -65.0298157, 64.3517227

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4521395, upper bound: 57.4979221
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5089681, upper bound: 57.5187778
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -10.1770201, 42.4380188, -51.2498398, 47.5348167
1: -11.2442751, 42.3229713, -12.9492474, 48.0443954, -59.2886696, 55.2722168
2: -11.0679474, 41.8824844, -12.7204800, 47.7395439, -58.8074799, 54.6029587
3: -19.1606712, 45.1869125, -21.9290562, 51.2056084, -70.3662796, 67.1159668
4: -17.8871288, 43.1344337, -20.4464874, 49.1660233, -67.0531387, 63.5809097

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5316536, upper bound: 57.5219302
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5316536, upper bound: 57.5512705
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -11.3360462, 47.0514069, -56.2089958, 50.2104721
1: -11.6959438, 44.0212021, -14.4060440, 53.2275314, -64.9234772, 58.4272461
2: -11.4683819, 43.6604576, -14.1231031, 53.0575676, -64.5259476, 57.7835579
3: -19.9659348, 46.8748016, -24.3241272, 56.5804291, -76.5463638, 71.1989288
4: -18.4506874, 44.9865723, -22.5729065, 54.6305351, -73.0812225, 67.5594788

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4788593, upper bound: 57.4582233
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4908181, upper bound: 57.5064693
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -11.9793425, 49.4708633, -58.2826843, 49.3371353
1: -11.2442751, 42.3229713, -15.2106190, 55.9559326, -67.2002029, 57.5335922
2: -11.0679474, 41.8824844, -14.9106045, 55.8369446, -66.9048920, 56.7930908
3: -19.1606712, 45.1869125, -25.6317139, 59.4555740, -78.6162415, 70.8186111
4: -17.8871288, 43.1344337, -23.7928352, 57.5261345, -75.4132385, 66.9272690

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5253590, upper bound: 57.5113879
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5342944, upper bound: 57.5374856
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -11.3360462, 47.0514069, -9.1575928, 38.8744278, -50.2104721, 56.2089958
1: -14.4060440, 53.2275314, -11.6959438, 44.0212021, -58.4272461, 64.9234772
2: -14.1231031, 53.0575676, -11.4683819, 43.6604576, -57.7835541, 64.5259476
3: -24.3241272, 56.5804291, -19.9659348, 46.8748016, -71.1989288, 76.5463638
4: -22.5729065, 54.6305351, -18.4506874, 44.9865723, -67.5594788, 73.0812225

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4582233, upper bound: 57.4788593
time: 0.51 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5064693, upper bound: 57.4908181
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -8.8118200, 37.3577957, -49.3371391, 58.2826843
1: -15.2106190, 55.9559326, -11.2442751, 42.3229713, -57.5335922, 67.2001953
2: -14.9106045, 55.8369446, -11.0679474, 41.8824844, -56.7930908, 66.9048920
3: -25.6317139, 59.4555740, -19.1606712, 45.1869125, -70.8186111, 78.6162415
4: -23.7928352, 57.5261345, -17.8871288, 43.1344337, -66.9272690, 75.4132462

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5113879, upper bound: 57.5253590
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5374856, upper bound: 57.5342944
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -11.3360462, 47.0514069, -11.0297852, 46.1416092, -57.4776535, 58.0811920
1: -14.4060440, 53.2275314, -14.0756989, 52.2037125, -66.6097565, 67.3032303
2: -14.1231031, 53.0575676, -13.7119274, 52.0922241, -66.2153244, 66.7694931
3: -24.3241272, 56.5804291, -23.8188839, 55.4197121, -79.7438354, 80.3993149
4: -22.5729065, 54.6305351, -21.8664837, 53.5777702, -76.1506729, 76.4970169

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4689596, upper bound: 57.4107255
time: 0.50 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4829711, upper bound: 57.4639132
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -11.9793425, 49.4708633, -10.4046993, 43.4960098, -55.4753494, 59.8755608
1: -15.2106190, 55.9559326, -13.2431650, 49.2144547, -64.4250717, 69.1990967
2: -14.9106045, 55.8369446, -12.9902172, 48.9800606, -63.8906631, 68.8271637
3: -25.6317139, 59.4555740, -22.4255047, 52.3740425, -78.0057449, 81.8810806
4: -23.7928352, 57.5261345, -20.8130035, 50.4081459, -74.2009735, 78.3391266

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4762997, upper bound: 57.4861621
time: 0.53 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4762997, upper bound: 57.5621587
time: 0.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.73 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4521395, upper bound: 57.4979221
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5089681, upper bound: 57.5187778
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5316536, upper bound: 57.5219302
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5316536, upper bound: 57.5512705
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4788593, upper bound: 57.4582233
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4908181, upper bound: 57.5064693
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5253590, upper bound: 57.5113879
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5342944, upper bound: 57.5374856
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4582233, upper bound: 57.4788593
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5064693, upper bound: 57.4908181
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5113879, upper bound: 57.5253590
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.5374856, upper bound: 57.5342944
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4689596, upper bound: 57.4107255
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4829711, upper bound: 57.4639132
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4762997, upper bound: 57.4861621
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -57.4762997, upper bound: 57.5621587

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -9.6038160, 40.2809982, -46.5373383, 38.2971268
1: -7.9915113, 32.5977592, -12.2323484, 45.6113548, -53.6028633, 44.8300972
2: -7.9890079, 31.8166008, -12.0197229, 45.2531128, -53.2421112, 43.8363152
3: -14.0460072, 34.9063377, -20.7653065, 48.6373138, -62.6833115, 55.6716423
4: -13.2832928, 32.6852341, -19.3651485, 46.5791321, -59.8624115, 52.0503845

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4355310, upper bound: 57.4515008
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4355310, upper bound: 57.4979221
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -8.5084524, 36.0848961, -43.5199585, 40.9692993
1: -9.5189266, 36.8422890, -10.8518114, 40.8862343, -50.4051590, 47.6940956
2: -9.4125261, 36.2255974, -10.6953497, 40.4268608, -49.8393860, 46.9209480
3: -16.3478165, 39.4466934, -18.4875679, 43.6790886, -60.0269051, 57.9342499
4: -15.2531500, 37.2985878, -17.2940769, 41.6261444, -56.8792953, 54.5926628

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5187778
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -9.1575928, 38.8744278, -47.6862488, 46.5153885
1: -11.2442751, 42.3229713, -11.6959438, 44.0212021, -55.2654762, 54.0189133
2: -11.0679474, 41.8824844, -11.4683819, 43.6604576, -54.7284012, 53.3508682
3: -19.1606712, 45.1869125, -19.9659348, 46.8748016, -66.0354691, 65.1528397
4: -17.8871288, 43.1344337, -18.4506874, 44.9865723, -62.8737030, 61.5851212

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4979221, upper bound: 57.4521395
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061431, upper bound: 57.5089681
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -8.8118200, 37.3577957, -46.1696167, 46.1696167
1: -11.2442751, 42.3229713, -11.2442751, 42.3229713, -53.5672455, 53.5672455
2: -11.0679474, 41.8824844, -11.0679474, 41.8824844, -52.9504280, 52.9504280
3: -19.1606712, 45.1869125, -19.1606712, 45.1869125, -64.3475800, 64.3475800
4: -17.8871288, 43.1344337, -17.8871288, 43.1344337, -61.0215607, 61.0215607

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4979221, upper bound: 57.4945006
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5187778, upper bound: 57.5326214
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.1575928, 38.8744278, -8.3103456, 36.4285011, -45.5860901, 47.1847649
1: -11.6959438, 44.0212021, -10.6107883, 41.2406464, -52.9365921, 54.6319847
2: -11.4683819, 43.6604576, -10.4886055, 40.7226562, -52.1910400, 54.1490555
3: -19.9659348, 46.8748016, -18.2756023, 43.9959564, -63.9618721, 65.1503983
4: -18.4506874, 44.9865723, -17.0999012, 41.8161430, -60.2668304, 62.0864716

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4190400, upper bound: 57.4354496
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4190400, upper bound: 57.4582233
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.9189520, 34.1508293, -9.3084240, 39.2523003, -47.1712418, 43.4592476
1: -10.1313725, 38.7070160, -11.8716488, 44.4490318, -54.5803986, 50.5786667
2: -9.9746494, 38.2284126, -11.6706486, 44.0363541, -54.0109978, 49.8990555
3: -17.3889561, 41.3090820, -20.0979748, 47.4316902, -64.8206482, 61.4070511
4: -16.1339893, 39.4080963, -18.6774368, 45.2791901, -61.4131775, 58.0855331

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4906624, upper bound: 57.5063255
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4908181, upper bound: 57.5064693
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -8.8355389, 38.4204979, -47.2323189, 46.1933327
1: -11.2442751, 42.3229713, -11.2725945, 43.4727173, -54.7169914, 53.5955658
2: -11.0679474, 41.8824844, -11.1283092, 43.0069008, -54.0748405, 53.0107956
3: -19.1606712, 45.1869125, -19.3558121, 46.3429070, -65.5035782, 64.5427170
4: -17.8871288, 43.1344337, -18.0807438, 44.1607170, -62.0478439, 61.2151794

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4593619, upper bound: 57.4082406
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4593619, upper bound: 57.4996187
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.7804146, 33.4704285, -9.8936510, 41.4662666, -49.2466812, 43.3640785
1: -9.9382153, 37.9656830, -12.6075249, 46.9325066, -56.8707199, 50.5732079
2: -9.8243971, 37.4091911, -12.3815975, 46.5865746, -56.4109726, 49.7907829
3: -17.0109787, 40.6094360, -21.2981529, 50.0432014, -67.0541763, 61.9075775
4: -15.9533205, 38.5370598, -19.7658882, 47.8902588, -63.8435783, 58.3029442

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4692414, upper bound: 57.4600438
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4692414, upper bound: 57.5326251
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -8.3103456, 36.4285011, -9.1575928, 38.8744278, -47.1847725, 45.5860901
1: -10.6107883, 41.2406464, -11.6959438, 44.0212021, -54.6319847, 52.9365921
2: -10.4886055, 40.7226562, -11.4683819, 43.6604576, -54.1490555, 52.1910400
3: -18.2756023, 43.9959564, -19.9659348, 46.8748016, -65.1504059, 63.9618645
4: -17.0999012, 41.8161430, -18.4506874, 44.9865723, -62.0864716, 60.2668304

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4354496, upper bound: 57.4190400
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4354496, upper bound: 57.4788593
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -9.3084240, 39.2523003, -7.9189520, 34.1508293, -43.4592476, 47.1712456
1: -11.8716488, 44.4490318, -10.1313725, 38.7070160, -50.5786667, 54.5803986
2: -11.6706486, 44.0363541, -9.9746494, 38.2284126, -49.8990555, 54.0109978
3: -20.0979748, 47.4316902, -17.3889561, 41.3090820, -61.4070473, 64.8206482
4: -18.6774368, 45.2791901, -16.1339893, 39.4080963, -58.0855331, 61.4131775

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5063255, upper bound: 57.4906624
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5064693, upper bound: 57.4908181
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -8.8355389, 38.4204979, -8.8118200, 37.3577957, -46.1933327, 47.2323189
1: -11.2725945, 43.4727173, -11.2442751, 42.3229713, -53.5955658, 54.7169914
2: -11.1283092, 43.0069008, -11.0679474, 41.8824844, -53.0107956, 54.0748405
3: -19.3558121, 46.3429070, -19.1606712, 45.1869125, -64.5427170, 65.5035782
4: -18.0807438, 44.1607170, -17.8871288, 43.1344337, -61.2151756, 62.0478439

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4056435, upper bound: 57.4593619
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4056435, upper bound: 57.5253590
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -9.8936510, 41.4662666, -7.7804146, 33.4704285, -43.3640785, 49.2466812
1: -12.6075249, 46.9325066, -9.9382153, 37.9656830, -50.5732079, 56.8707199
2: -12.3815975, 46.5865746, -9.8243971, 37.4091911, -49.7907829, 56.4109726
3: -21.2981529, 50.0432014, -17.0109787, 40.6094360, -61.9075737, 67.0541611
4: -19.7658882, 47.8902588, -15.9533205, 38.5370598, -58.3029442, 63.8435783

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4583964, upper bound: 57.4692414
time: 0.57 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4583964, upper bound: 57.5342944
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -11.3360462, 47.0514069, -8.0627384, 35.7259407, -47.0619888, 55.1141434
1: -14.4060440, 53.2275314, -10.3429604, 40.4529800, -54.8590202, 63.5704918
2: -14.1231031, 53.0575676, -10.1579933, 39.9914627, -54.1145668, 63.2155609
3: -24.3241272, 56.5804291, -17.8792992, 43.0726242, -67.3967514, 74.4597321
4: -22.5729065, 54.6305351, -16.5315685, 41.0846443, -63.6575508, 71.1620941

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4331465, upper bound: 57.3947674
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4331465, upper bound: 57.4107255
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -9.8670855, 41.4268723, -9.0433788, 38.5837364, -48.4508209, 50.4702530
1: -12.5476713, 46.8760223, -11.5689020, 43.6965904, -56.2442627, 58.4449234
2: -12.3394318, 46.5901794, -11.3229685, 43.3287659, -55.6681976, 57.9131393
3: -21.2743568, 49.9116249, -19.6716652, 46.5399094, -67.8142700, 69.5832901
4: -19.7754040, 47.9410820, -18.0751343, 44.5710564, -64.3464584, 66.0162201

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4505198, upper bound: 57.4505198
time: 0.52 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4505198, upper bound: 57.4639132
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -10.4046993, 43.4960098, -54.5257912, 56.5463028
1: -14.0756989, 52.2037125, -13.2431650, 49.2144547, -63.2901535, 65.4468765
2: -13.7119274, 52.0922241, -12.9902172, 48.9800606, -62.6919861, 65.0824432
3: -23.8188839, 55.4197121, -22.4255047, 52.3740425, -76.1929245, 77.8452148
4: -21.8664837, 53.5777702, -20.8130035, 50.4081459, -72.2746277, 74.3907700

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3947674, upper bound: 57.4593619
time: 0.50 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4505198, upper bound: 57.4692414
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -10.4046993, 43.4960098, -53.9007034, 53.9006996
1: -13.2431650, 49.2144547, -13.2431650, 49.2144547, -62.4576187, 62.4576187
2: -12.9902172, 48.9800606, -12.9902172, 48.9800606, -61.9702759, 61.9702644
3: -22.4255047, 52.3740425, -22.4255047, 52.3740425, -74.7995453, 74.7995453
4: -20.8130035, 50.4081459, -20.8130035, 50.4081459, -71.2211304, 71.2211304

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3947674, upper bound: 57.5329379
time: 0.51 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4505198, upper bound: 57.5399641
time: 0.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.77 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4355310, upper bound: 57.4515008
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4355310, upper bound: 57.4979221
NS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
NS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5187778
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4979221, upper bound: 57.4521395
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.5061431, upper bound: 57.5089681
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4979221, upper bound: 57.4945006
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.5187778, upper bound: 57.5326214
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4190400, upper bound: 57.4354496
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4190400, upper bound: 57.4582233
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4906624, upper bound: 57.5063255
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4908181, upper bound: 57.5064693
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4593619, upper bound: 57.4082406
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4593619, upper bound: 57.4996187
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4692414, upper bound: 57.4600438
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4692414, upper bound: 57.5326251
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4354496, upper bound: 57.4190400
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4354496, upper bound: 57.4788593
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.5063255, upper bound: 57.4906624
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.5064693, upper bound: 57.4908181
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4056435, upper bound: 57.4593619
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4056435, upper bound: 57.5253590
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4583964, upper bound: 57.4692414
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4583964, upper bound: 57.5342944
NS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4331465, upper bound: 57.3947674
NS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4331465, upper bound: 57.4107255
NS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4505198, upper bound: 57.4505198
NS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4505198, upper bound: 57.4639132
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.3947674, upper bound: 57.4593619
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4505198, upper bound: 57.4692414
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.3947674, upper bound: 57.5329379
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 0, lower bound: -57.4505198, upper bound: 57.5399641

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -6.6542845, 29.9349804, -36.1913261, 35.3475914
1: -7.9915113, 32.5977592, -8.4881611, 34.0101814, -42.0016899, 41.0859222
2: -7.9890079, 31.8166008, -8.4828358, 33.2241821, -41.2131805, 40.2994385
3: -14.0460072, 34.9063377, -14.7928667, 36.4859810, -50.5319901, 49.6992035
4: -13.2832928, 32.6852341, -14.1002512, 34.1036415, -47.3869286, 46.7854843

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4292205
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4515008
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -7.9464331, 34.1318932, -40.3882370, 36.6397476
1: -7.9915113, 32.5977592, -10.1651115, 38.7476234, -46.7391319, 42.7628632
2: -7.9890079, 31.8166008, -10.0406456, 38.1110229, -46.1000290, 41.8572464
3: -14.0460072, 34.9063377, -17.3263245, 41.5232506, -55.5692596, 52.2326622
4: -13.2832928, 32.6852341, -16.2702694, 39.2173157, -52.5006065, 48.9555054

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4227716, upper bound: 57.4978013
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4355310, upper bound: 57.4979221
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -7.7420950, 33.3329544, -40.7680168, 40.2029419
1: -9.5189266, 36.8422890, -9.8999453, 37.7919540, -47.3108826, 46.7422333
2: -9.4125261, 36.2255974, -9.7552605, 37.2998810, -46.7124062, 45.9808578
3: -16.3478165, 39.4466934, -16.9821148, 40.3600883, -56.7079048, 56.4287987
4: -15.2531500, 37.2985878, -15.7892637, 38.4518776, -53.7050247, 53.0878525

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4494298
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -7.7804146, 33.4704285, -40.9054909, 40.2412605
1: -9.5189266, 36.8422890, -9.9382153, 37.9656830, -47.4846115, 46.7805023
2: -9.4125261, 36.2255974, -9.8243971, 37.4091911, -46.8217163, 46.0499954
3: -16.3478165, 39.4466934, -17.0109787, 40.6094360, -56.9572525, 56.4576645
4: -15.2531500, 37.2985878, -15.9533205, 38.5370598, -53.7902107, 53.2519073

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4965629, upper bound: 57.5112629
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5046515, upper bound: 57.5161700
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -6.2563448, 28.6933155, -37.5051346, 43.6141396
1: -11.2442751, 42.3229713, -7.9915113, 32.5977592, -43.8420258, 50.3144836
2: -11.0679474, 41.8824844, -7.9890079, 31.8166008, -42.8845406, 49.8714867
3: -19.1606712, 45.1869125, -14.0460072, 34.9063377, -54.0670052, 59.2329178
4: -17.8871288, 43.1344337, -13.2832928, 32.6852341, -50.5723648, 56.4177208

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4355310
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4521395
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -7.7804146, 33.4704285, -7.4350624, 32.4608459, -40.2412605, 40.9054909
1: -9.9382153, 37.9656830, -9.5189266, 36.8422890, -46.7805023, 47.4846115
2: -9.8243971, 37.4091911, -9.4125261, 36.2255974, -46.0499954, 46.8217163
3: -17.0109787, 40.6094360, -16.3478165, 39.4466934, -56.4576645, 56.9572525
4: -15.9533205, 38.5370598, -15.2531500, 37.2985878, -53.2519073, 53.7902107

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5112629, upper bound: 57.5029705
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5161699, upper bound: 57.5073595
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -5.9096947, 27.3499146, -36.1617317, 43.2674904
1: -11.2442751, 42.3229713, -7.5164185, 31.1457329, -42.3900032, 49.8393898
2: -11.0679474, 41.8824844, -7.6001306, 30.2378502, -41.3057938, 49.4826164
3: -19.1606712, 45.1869125, -13.2140646, 33.4282455, -52.5889130, 58.4009743
4: -17.8871288, 43.1344337, -12.7368975, 30.9877415, -48.8748703, 55.8713303

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4814832, upper bound: 57.4814832
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4945006
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -7.7804146, 33.4704285, -7.3467827, 32.0613708, -39.8417816, 40.8172035
1: -9.9382153, 37.9656830, -9.3942413, 36.4329567, -46.3711662, 47.3599243
2: -9.8243971, 37.4091911, -9.3245058, 35.7049446, -45.5293388, 46.7336960
3: -17.0109787, 40.6094360, -16.1041088, 39.0988007, -56.1097717, 56.7135468
4: -15.9533205, 38.5370598, -15.1873417, 36.7418594, -52.6951790, 53.7244034

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -8.3103456, 36.4285011, -42.6848450, 37.0036621
1: -7.9915113, 32.5977592, -10.6107883, 41.2406464, -49.2321587, 43.2085381
2: -7.9890079, 31.8166008, -10.4886055, 40.7226562, -48.7116623, 42.3051949
3: -14.0460072, 34.9063377, -18.2756023, 43.9959564, -58.0419579, 53.1819382
4: -13.2832928, 32.6852341, -17.0999012, 41.8161430, -55.0994339, 49.7851334

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3922841, upper bound: 57.3852977
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4354496
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -8.3103456, 36.4285011, -43.8635635, 40.7711868
1: -9.5189266, 36.8422890, -10.6107883, 41.2406464, -50.7595749, 47.4530716
2: -9.4125261, 36.2255974, -10.4886055, 40.7226562, -50.1351814, 46.7141991
3: -16.3478165, 39.4466934, -18.2756023, 43.9959564, -60.3437691, 57.7222862
4: -15.2531500, 37.2985878, -17.0999012, 41.8161430, -57.0692940, 54.3984909

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4144497, upper bound: 57.4385264
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4175516, upper bound: 57.4574595
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -7.3741217, 32.0932846, -7.3768864, 32.1902809, -39.5643997, 39.4701614
1: -9.4388428, 36.4032402, -9.4218798, 36.5786400, -46.0174828, 45.8251190
2: -9.3183212, 35.8604546, -9.3372574, 35.9144287, -45.2327499, 45.1977081
3: -16.2560253, 38.8825417, -16.1180038, 39.1414795, -55.3974991, 55.0005455
4: -15.1103363, 36.9993744, -15.1752281, 36.9083405, -52.0186653, 52.1746025

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4188573, upper bound: 57.4811165
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4188573, upper bound: 57.5001644
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -7.7393336, 33.4550667, -8.7309418, 37.0282707, -44.7676010, 42.1860085
1: -9.9046087, 37.9284363, -11.1410065, 41.9653206, -51.8699303, 49.0694389
2: -9.7569246, 37.4305687, -10.9694843, 41.4766235, -51.2335434, 48.4000549
3: -17.0155411, 40.4901886, -18.9007130, 44.8257675, -61.8413048, 59.3908997
4: -15.7923479, 38.5961914, -17.5931549, 42.6702728, -58.4626122, 56.1893463

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4635848, upper bound: 57.4583964
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4635848, upper bound: 57.5064693
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -8.0627384, 35.7259407, -44.5377617, 45.4205322
1: -11.2442751, 42.3229713, -10.3429604, 40.4529800, -51.6972504, 52.6659317
2: -11.0679474, 41.8824844, -10.1579933, 39.9914627, -51.0594063, 52.0404739
3: -19.1606712, 45.1869125, -17.8792992, 43.0726242, -62.2332954, 63.0662117
4: -17.8871288, 43.1344337, -16.5315685, 41.0846443, -58.9717712, 59.6659966

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4072148, upper bound: 57.3896511
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4082406
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.8118200, 37.3577957, -7.4656463, 33.3557091, -42.1675301, 44.8234406
1: -11.2442751, 42.3229713, -9.5422745, 37.8093300, -49.0536041, 51.8652420
2: -11.0679474, 41.8824844, -9.4745121, 37.1891479, -48.2570953, 51.3569946
3: -19.1606712, 45.1869125, -16.5416107, 40.3914528, -59.5521240, 61.7285194
4: -17.8871288, 43.1344337, -15.5500689, 38.1762428, -56.0633698, 58.6845016

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4814830
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4996189
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -7.7804146, 33.4704285, -8.8100138, 37.5939674, -45.3743820, 42.2804375
1: -9.9382153, 37.9656830, -11.2688313, 42.5857544, -52.5239601, 49.2345123
2: -9.8243971, 37.4091911, -11.0355730, 42.1981087, -52.0225067, 48.4447632
3: -17.0109787, 40.6094360, -19.1602993, 45.3824959, -62.3934593, 59.7697258
4: -15.9533205, 38.5370598, -17.6283436, 43.3953476, -59.3486671, 56.1653938

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4639549, upper bound: 57.4545997
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4662070, upper bound: 57.4573161
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -7.7804146, 33.4704285, -8.7081556, 37.1425133, -44.9229279, 42.1785851
1: -9.9382153, 37.9656830, -11.1119413, 42.0899849, -52.0281982, 49.0776253
2: -9.8243971, 37.4091911, -10.9512491, 41.5937920, -51.4181862, 48.3604393
3: -17.0109787, 40.6094360, -18.8729000, 44.9509163, -61.9618874, 59.4823341
4: -15.9533205, 38.5370598, -17.5746422, 42.7677650, -58.7210846, 56.1117020

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4639549, upper bound: 57.5249882
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4662070, upper bound: 57.5306095
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.3103456, 36.4285011, -6.2563448, 28.6933155, -37.0036545, 42.6848450
1: -10.6107883, 41.2406464, -7.9915113, 32.5977592, -43.2085381, 49.2321587
2: -10.4886055, 40.7226562, -7.9890079, 31.8166008, -42.3051949, 48.7116623
3: -18.2756023, 43.9959564, -14.0460072, 34.9063377, -53.1819344, 58.0419617
4: -17.0999012, 41.8161430, -13.2832928, 32.6852341, -49.7851334, 55.0994339

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.3922841
time: 0.46 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4190400
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.3103456, 36.4285011, -7.4350624, 32.4608459, -40.7711868, 43.8635635
1: -10.6107883, 41.2406464, -9.5189266, 36.8422890, -47.4530716, 50.7595749
2: -10.4886055, 40.7226562, -9.4125261, 36.2255974, -46.7141991, 50.1351814
3: -18.2756023, 43.9959564, -16.3478165, 39.4466934, -57.7222862, 60.3437653
4: -17.0999012, 41.8161430, -15.2531500, 37.2985878, -54.3984909, 57.0692940

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4141729, upper bound: 57.4741026
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4343078, upper bound: 57.4771422
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -7.3768864, 32.1902809, -7.3741217, 32.0932846, -39.4701653, 39.5643997
1: -9.4218798, 36.5786400, -9.4388428, 36.4032402, -45.8251190, 46.0174828
2: -9.3372574, 35.9144287, -9.3183212, 35.8604546, -45.1977081, 45.2327499
3: -16.1180038, 39.1414795, -16.2560253, 38.8825417, -55.0005417, 55.3975029
4: -15.1752281, 36.9083405, -15.1103363, 36.9993744, -52.1746025, 52.0186653

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4811164, upper bound: 57.4222640
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4811165, upper bound: 57.4906624
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -8.7309418, 37.0282707, -7.7393336, 33.4550667, -42.1860085, 44.7676010
1: -11.1410065, 41.9653206, -9.9046087, 37.9284363, -49.0694389, 51.8699303
2: -10.9694843, 41.4766235, -9.7569246, 37.4305687, -48.4000549, 51.2335434
3: -18.9007130, 44.8257675, -17.0155411, 40.4901886, -59.3908997, 61.8413048
4: -17.5931549, 42.6702728, -15.7923479, 38.5961914, -56.1893463, 58.4626122

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4583964, upper bound: 57.4635848
time: 0.60 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4583964, upper bound: 57.4908181
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -8.8118200, 37.3577957, -45.4205322, 44.5377617
1: -10.3429604, 40.4529800, -11.2442751, 42.3229713, -52.6659317, 51.6972504
2: -10.1579933, 39.9914627, -11.0679474, 41.8824844, -52.0404739, 51.0594063
3: -17.8792992, 43.0726242, -19.1606712, 45.1869125, -63.0662003, 62.2332954
4: -16.5315685, 41.0846443, -17.8871288, 43.1344337, -59.6659966, 58.9717712

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4072148
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4593619
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -7.4656463, 33.3557091, -8.8118200, 37.3577957, -44.8234406, 42.1675301
1: -9.5422745, 37.8093300, -11.2442751, 42.3229713, -51.8652420, 49.0536041
2: -9.4745121, 37.1891479, -11.0679474, 41.8824844, -51.3569946, 48.2570915
3: -16.5416107, 40.3914528, -19.1606712, 45.1869125, -61.7285156, 59.5521240
4: -15.5500689, 38.1762428, -17.8871288, 43.1344337, -58.6845016, 56.0633698

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4863925
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.5253590
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -8.8100138, 37.5939674, -7.7804146, 33.4704285, -42.2804413, 45.3743820
1: -11.2688313, 42.5857544, -9.9382153, 37.9656830, -49.2345123, 52.5239601
2: -11.0355730, 42.1981087, -9.8243971, 37.4091911, -48.4447632, 52.0225067
3: -19.1602993, 45.3824959, -17.0109787, 40.6094360, -59.7697258, 62.3934593
4: -17.6283436, 43.3953476, -15.9533205, 38.5370598, -56.1653938, 59.3486671

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4529330, upper bound: 57.4639549
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4556863, upper bound: 57.4662070
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.7081556, 37.1425133, -7.7804146, 33.4704285, -42.1785812, 44.9229279
1: -11.1119413, 42.0899849, -9.9382153, 37.9656830, -49.0776253, 52.0281982
2: -10.9512491, 41.5937920, -9.8243971, 37.4091911, -48.3604393, 51.4181862
3: -18.8729000, 44.9509163, -17.0109787, 40.6094360, -59.4823341, 61.9618874
4: -17.5746422, 42.7677650, -15.9533205, 38.5370598, -56.1117020, 58.7210846

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4529330, upper bound: 57.5238700
time: 0.62 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4556863, upper bound: 57.5322411
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -11.0297852, 46.1416092, -8.0627384, 35.7259407, -46.7557259, 54.2043457
1: -14.0756989, 52.2037125, -10.3429604, 40.4529800, -54.5286789, 62.5466728
2: -13.7119274, 52.0922241, -10.1579933, 39.9914627, -53.7033882, 62.2502174
3: -23.8188839, 55.4197121, -17.8792992, 43.0726242, -66.8915100, 73.2990112
4: -21.8664837, 53.5777702, -16.5315685, 41.0846443, -62.9511261, 70.1093292

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.3773941
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.3947674
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -10.4046993, 43.4960098, -8.0627384, 35.7259407, -46.1306381, 51.5587463
1: -13.2431650, 49.2144547, -10.3429604, 40.4529800, -53.6961403, 59.5574150
2: -12.9902172, 48.9800606, -10.1579933, 39.9914627, -52.9816780, 59.1380539
3: -22.4255047, 52.3740425, -17.8792992, 43.0726242, -65.4981308, 70.2533264
4: -20.8130035, 50.4081459, -16.5315685, 41.0846443, -61.8976440, 66.9396896

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.3944190
time: 0.49 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4107255
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -9.6458893, 40.8463058, -9.0433788, 38.5837364, -48.2296257, 49.8896828
1: -12.3220186, 46.2224998, -11.5689020, 43.6965904, -56.0186043, 57.7914009
2: -12.0382032, 46.0077515, -11.3229685, 43.3287659, -55.3669701, 57.3307190
3: -20.9314880, 49.1433983, -19.6716652, 46.5399094, -67.4713974, 68.8150635
4: -19.2438354, 47.3377991, -18.0751343, 44.5710564, -63.8148880, 65.4129181

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4331465
time: 0.50 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4504625
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -9.1056776, 38.5635071, -9.0433788, 38.5837364, -47.6894035, 47.6068878
1: -11.6022539, 43.6453171, -11.5689020, 43.6965904, -55.2988396, 55.2142181
2: -11.4254990, 43.3039856, -11.3229685, 43.3287659, -54.7542648, 54.6269531
3: -19.7298203, 46.5255547, -19.6716652, 46.5399094, -66.2697296, 66.1972198
4: -18.3488369, 44.5894966, -18.0751343, 44.5710564, -62.9198837, 62.6646309

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4477043
time: 0.58 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4639132
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -10.4046993, 43.4960098, -51.5587463, 46.1306381
1: -10.3429604, 40.4529800, -13.2431650, 49.2144547, -59.5574150, 53.6961403
2: -10.1579933, 39.9914627, -12.9902172, 48.9800606, -59.1380539, 52.9816780
3: -17.8792992, 43.0726242, -22.4255047, 52.3740425, -70.2533340, 65.4981308
4: -16.5315685, 41.0846443, -20.8130035, 50.4081459, -66.9396896, 61.8976479

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3896511, upper bound: 57.4072148
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3896511, upper bound: 57.4593619
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -9.0433788, 38.5837364, -9.1056776, 38.5635071, -47.6068878, 47.6893997
1: -11.5689020, 43.6965904, -11.6022539, 43.6453171, -55.2142181, 55.2988434
2: -11.3229685, 43.3287659, -11.4254990, 43.3039856, -54.6269531, 54.7542648
3: -19.6716652, 46.5399094, -19.7298203, 46.5255547, -66.1972198, 66.2697296
4: -18.0751343, 44.5710564, -18.3488369, 44.5894966, -62.6646309, 62.9198875

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4401649, upper bound: 57.4125992
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4401649, upper bound: 57.4692414
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -7.4656463, 33.3557091, -10.4046993, 43.4960098, -50.9616547, 43.7604027
1: -9.5422745, 37.8093300, -13.2431650, 49.2144547, -58.7567291, 51.0524940
2: -9.4745121, 37.1891479, -12.9902172, 48.9800606, -58.4545746, 50.1793671
3: -16.5416107, 40.3914528, -22.4255047, 52.3740425, -68.9156418, 62.8169556
4: -15.5500689, 38.1762428, -20.8130035, 50.4081459, -65.9582062, 58.9892349

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5155761, upper bound: 57.5278145
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5048539, upper bound: 57.5244702
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.7081556, 37.1425133, -9.1056776, 38.5635071, -47.2716637, 46.2481842
1: -11.1119413, 42.0899849, -11.6022539, 43.6453171, -54.7572594, 53.6922379
2: -10.9512491, 41.5937920, -11.4254990, 43.3039856, -54.2552338, 53.0192909
3: -18.8729000, 44.9509163, -19.7298203, 46.5255547, -65.3984528, 64.6807404
4: -17.5746422, 42.7677650, -18.3488369, 44.5894966, -62.1641388, 61.1166000

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5381886, upper bound: 57.5361746
time: 0.58 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5348344, upper bound: 57.5348344
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.05 seconds
NS_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4292205
NS_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4515008
NS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4227716, upper bound: 57.4978013
NS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4355310, upper bound: 57.4979221
NS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4494298
NS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
NS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4965629, upper bound: 57.5112629
NS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5046515, upper bound: 57.5161700
NS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4355310
NS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4521395
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5112629, upper bound: 57.5029705
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5161699, upper bound: 57.5073595
NS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4814832, upper bound: 57.4814832
NS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4945006
NS_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
NS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3922841, upper bound: 57.3852977
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4354496
NS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4144497, upper bound: 57.4385264
NS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4175516, upper bound: 57.4574595
NS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4188573, upper bound: 57.4811165
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4188573, upper bound: 57.5001644
NS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4635848, upper bound: 57.4583964
NS_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4635848, upper bound: 57.5064693
NS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4072148, upper bound: 57.3896511
NS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4082406
NS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4814830
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4996189
NS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4639549, upper bound: 57.4545997
NS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4662070, upper bound: 57.4573161
NS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4639549, upper bound: 57.5249882
NS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4662070, upper bound: 57.5306095
NS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3852977, upper bound: 57.3922841
NS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4190400
NS_A2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4141729, upper bound: 57.4741026
NS_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4343078, upper bound: 57.4771422
NS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4811164, upper bound: 57.4222640
NS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4811165, upper bound: 57.4906624
NS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4583964, upper bound: 57.4635848
NS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4583964, upper bound: 57.4908181
NS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4072148
NS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4593619
NS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4863925
NS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3852977, upper bound: 57.5253590
NS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4529330, upper bound: 57.4639549
NS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4556863, upper bound: 57.4662070
NS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4529330, upper bound: 57.5238700
NS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4556863, upper bound: 57.5322411
NS_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.3773941
NS_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.3947674
NS_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.3944190
NS_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4107255
NS_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4331465
NS_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4504625
NS_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4477043
NS_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3773941, upper bound: 57.4639132
NS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3896511, upper bound: 57.4072148
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.3896511, upper bound: 57.4593619
NS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4401649, upper bound: 57.4125992
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.4401649, upper bound: 57.4692414
NS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5155761, upper bound: 57.5278145
NS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5048539, upper bound: 57.5244702
NS_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5381886, upper bound: 57.5361746
NS_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 0, lower bound: -57.5348344, upper bound: 57.5348344

## BFS NS instance: NS_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -6.2563448, 28.6933155, -34.9496574, 34.9496574
1: -7.9915113, 32.5977592, -7.9915113, 32.5977592, -40.5892639, 40.5892639
2: -7.9890079, 31.8166008, -7.9890079, 31.8166008, -39.8055954, 39.8055954
3: -14.0460072, 34.9063377, -14.0460072, 34.9063377, -48.9523468, 48.9523468
4: -13.2832928, 32.6852341, -13.2832928, 32.6852341, -45.9685287, 45.9685287

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3535846, upper bound: 57.3859768
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3333625, upper bound: 57.3333625
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -5.9096947, 27.3499146, -33.6062546, 34.6030121
1: -7.9915113, 32.5977592, -7.5164185, 31.1457329, -39.1372375, 40.1141777
2: -7.9890079, 31.8166008, -7.6001306, 30.2378502, -38.2268524, 39.4167213
3: -14.0460072, 34.9063377, -13.2140646, 33.4282455, -47.4742508, 48.1204033
4: -13.2832928, 32.6852341, -12.7368975, 30.9877415, -44.2710266, 45.4221306

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3859768, upper bound: 57.4075307
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3333625, upper bound: 57.3779775
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5.7156672, 26.6757584, -6.4726439, 28.7651939, -34.4808617, 33.1484032
1: -7.2887216, 30.3599243, -8.2765684, 32.7731819, -40.0619049, 38.6364861
2: -7.3479490, 29.5131912, -8.2744131, 31.9286728, -39.2766228, 37.7875900
3: -12.8840027, 32.5253181, -14.2483788, 35.2516251, -48.1356277, 46.7736931
4: -12.2607584, 30.3138466, -13.6550293, 32.8263893, -45.0871468, 43.9688759

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4031152, upper bound: 57.4704486
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4426372, upper bound: 57.4955818
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.0809369, 28.0204182, -7.3182898, 31.7754841, -37.8564186, 35.3387032
1: -7.7656422, 31.8482037, -9.3631554, 36.1108742, -43.8765068, 41.2113571
2: -7.7789221, 31.0486145, -9.2831860, 35.3949852, -43.1739082, 40.3318024
3: -13.6708202, 34.1138191, -16.0260944, 38.7580681, -52.4288864, 50.1399155
4: -12.9492245, 31.8974686, -15.1050768, 36.4438934, -49.3931160, 47.0025444

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4859338
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4979221
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -5.8050623, 26.8103561, -34.2454185, 38.2659073
1: -9.5189266, 36.8422890, -7.4013424, 30.4969997, -40.0159264, 44.2436295
2: -9.4125261, 36.2255974, -7.4495697, 29.6914902, -39.1040154, 43.6751671
3: -16.3478165, 39.4466934, -13.0302229, 32.7092361, -49.0570488, 52.4769096
4: -15.2531500, 37.2985878, -12.4200859, 30.4931049, -45.7462540, 49.7186699

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4540048, upper bound: 57.3776547
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4113590, upper bound: 57.3538622
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -7.2947378, 31.8497753, -39.2848358, 39.7555771
1: -9.5189266, 36.8422890, -9.3346424, 36.1582756, -45.6772003, 46.1769333
2: -9.4125261, 36.2255974, -9.2406588, 35.5294800, -44.9420052, 45.4662552
3: -16.3478165, 39.4466934, -16.0341167, 38.7422409, -55.0900574, 55.4808006
4: -15.2531500, 37.2985878, -14.9874439, 36.5850677, -51.8382187, 52.2860298

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4019974, upper bound: 57.4651135
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4113592, upper bound: 57.4465134
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -7.2590618, 31.7949753, -6.9876323, 30.5188522, -37.7779160, 38.7826080
1: -9.2938080, 36.0943527, -8.9284620, 34.6611366, -43.9549446, 45.0228157
2: -9.1984081, 35.4583359, -8.8685627, 33.9844780, -43.1828842, 44.3268929
3: -15.9812050, 38.6635780, -15.3430519, 37.1408958, -53.1221008, 54.0066261
4: -14.9267025, 36.5053978, -14.4918509, 34.9908066, -49.9175072, 50.9972458

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4807931, upper bound: 57.4787224
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4080314, upper bound: 57.4485118
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -7.4350624, 32.4608459, -7.1687813, 31.2684269, -38.7034912, 39.6296272
1: -9.5189266, 36.8422890, -9.1591225, 35.5011978, -45.0201263, 46.0014038
2: -9.4125261, 36.2255974, -9.0866833, 34.8562317, -44.2687569, 45.3122787
3: -16.3478165, 39.4466934, -15.7503662, 38.0228539, -54.3706703, 55.1970558
4: -15.2531500, 37.2985878, -14.8445187, 35.8955460, -51.1486931, 52.1431046

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4910985, upper bound: 57.4707716
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4910986, upper bound: 57.5146976
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -5.9096947, 27.3499146, -6.2563448, 28.6933155, -34.6030121, 33.6062508
1: -7.5164185, 31.1457329, -7.9915113, 32.5977592, -40.1141777, 39.1372375
2: -7.6001306, 30.2378502, -7.9890079, 31.8166008, -39.4167252, 38.2268524
3: -13.2140646, 33.4282455, -14.0460072, 34.9063377, -48.1204033, 47.4742508
4: -12.7368975, 30.9877415, -13.2832928, 32.6852341, -45.4221306, 44.2710266

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4075307, upper bound: 57.4022774
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3779775, upper bound: 57.3513281
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3467827, 32.0613708, -6.2563448, 28.6933155, -36.0400887, 38.3177109
1: -9.3942413, 36.4329567, -7.9915113, 32.5977592, -41.9919930, 44.4244652
2: -9.3245058, 35.7049446, -7.9890079, 31.8166008, -41.1411057, 43.6939430
3: -16.1041088, 39.0988007, -14.0460072, 34.9063377, -51.0104446, 53.1448059
4: -15.1873417, 36.7418594, -13.2832928, 32.6852341, -47.8725739, 50.0251541

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4354261, upper bound: 57.4446294
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4521395
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -6.9876323, 30.5188522, -7.2590618, 31.7949753, -38.7826080, 37.7779160
1: -8.9284620, 34.6611366, -9.2938080, 36.0943527, -45.0228157, 43.9549446
2: -8.8685627, 33.9844780, -9.1984081, 35.4583359, -44.3268967, 43.1828842
3: -15.3430519, 37.1408958, -15.9812050, 38.6635780, -54.0066261, 53.1221008
4: -14.4918509, 34.9908066, -14.9267025, 36.5053978, -50.9972458, 49.9175072

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4787224, upper bound: 57.4807931
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4485118, upper bound: 57.4080314
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -7.1687813, 31.2684269, -7.4350624, 32.4608459, -39.6296272, 38.7034912
1: -9.1591225, 35.5011978, -9.5189266, 36.8422890, -46.0014038, 45.0201263
2: -9.0866833, 34.8562317, -9.4125261, 36.2255974, -45.3122787, 44.2687569
3: -15.7503662, 38.0228539, -16.3478165, 39.4466934, -55.1970596, 54.3706703
4: -14.8445187, 35.8955460, -15.2531500, 37.2985878, -52.1431046, 51.1486931

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494365, upper bound: 57.4910985
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494365, upper bound: 57.5073460
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9096947, 27.3499146, -5.9096947, 27.3499146, -33.2596054, 33.2596054
1: -7.5164185, 31.1457329, -7.5164185, 31.1457329, -38.6621513, 38.6621513
2: -7.6001306, 30.2378502, -7.6001306, 30.2378502, -37.8379784, 37.8379784
3: -13.2140646, 33.4282455, -13.2140646, 33.4282455, -46.6423073, 46.6423073
4: -12.7368975, 30.9877415, -12.7368975, 30.9877415, -43.7246361, 43.7246361

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4557617, upper bound: 57.4282429
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4164057, upper bound: 57.4164057
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -7.3467827, 32.0613708, -5.9096947, 27.3499146, -34.6966782, 37.9710617
1: -9.3942413, 36.4329567, -7.5164185, 31.1457329, -40.5399704, 43.9493752
2: -9.3245058, 35.7049446, -7.6001306, 30.2378502, -39.5623550, 43.3050690
3: -16.1041088, 39.0988007, -13.2140646, 33.4282455, -49.5323563, 52.3128586
4: -15.1873417, 36.7418594, -12.7368975, 30.9877415, -46.1750832, 49.4787560

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4354261, upper bound: 57.4945006
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4945006
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -7.2643309, 31.5176945, -5.8067570, 26.4507256, -33.7150497, 37.3244514
1: -9.2837849, 35.7849121, -7.4054251, 30.2039413, -39.4877205, 43.1903381
2: -9.2002258, 35.1604843, -7.4816475, 29.2646160, -38.4648438, 42.6421318
3: -15.9407425, 38.3115158, -12.8401928, 32.5236092, -48.4643517, 51.1517105
4: -14.9800558, 36.2488327, -12.4464569, 30.0448952, -45.0249443, 48.6952896

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5151404, upper bound: 57.5111715
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5315038, upper bound: 57.5316335
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -7.5984273, 32.7688103, -6.7064815, 29.6760235, -37.2744522, 39.4752884
1: -9.7078514, 37.1806908, -8.5756721, 33.7746735, -43.4825249, 45.7563591
2: -9.6031847, 36.6030998, -8.5574265, 32.9536819, -42.5568657, 45.1605263
3: -16.6322136, 39.7835655, -14.7597656, 36.2993317, -52.9315453, 54.5433311
4: -15.6066999, 37.7166176, -14.0071154, 33.9241180, -49.5308189, 51.7237320

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -7.6970072, 34.3913345, -40.6476784, 36.3903236
1: -7.9915113, 32.5977592, -9.8640242, 38.9474487, -46.9389572, 42.4617767
2: -7.9890079, 31.8166008, -9.7420282, 38.4363785, -46.4253769, 41.5586205
3: -14.0460072, 34.9063377, -17.1286697, 41.5261650, -55.5721741, 52.0349998
4: -13.2832928, 32.6852341, -15.9208412, 39.5123138, -52.7955971, 48.6060753

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3447240, upper bound: 57.3643755
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3327170, upper bound: 57.3378868
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -7.4656463, 33.3557091, -39.6120529, 36.1589584
1: -7.9915113, 32.5977592, -9.5422745, 37.8093300, -45.8008423, 42.1400223
2: -7.9890079, 31.8166008, -9.4745121, 37.1891479, -45.1781540, 41.2911034
3: -14.0460072, 34.9063377, -16.5416107, 40.3914528, -54.4374619, 51.4479485
4: -13.2832928, 32.6852341, -15.5500689, 38.1762428, -51.4595337, 48.2353020

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3812309, upper bound: 57.4167830
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3327170, upper bound: 57.3838189
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -6.5591278, 30.0860119, -7.2943954, 32.5972137, -39.1563377, 37.3804092
1: -8.4003382, 34.2025681, -9.3045883, 36.9583969, -45.3587341, 43.5071564
2: -8.3580723, 33.4477882, -9.2544870, 36.2896118, -44.6476822, 42.7022743
3: -14.7368250, 36.5724945, -16.1635914, 39.5124702, -54.2492943, 52.7360840
4: -13.8810635, 34.3726807, -15.2678089, 37.2344894, -51.1155548, 49.6404800

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244031, upper bound: 57.3657514
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244031, upper bound: 57.4385264
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -7.0722103, 31.2705612, -8.3103456, 36.4285011, -43.5007095, 39.5809021
1: -9.0512457, 35.5169983, -10.6107883, 41.2406464, -50.2918930, 46.1277809
2: -8.9803925, 34.8309250, -10.4886055, 40.7226562, -49.7030487, 45.3195229
3: -15.6104374, 38.0437126, -18.2756023, 43.9959564, -59.6063805, 56.3193092
4: -14.6354523, 35.8261719, -17.0999012, 41.8161430, -56.4515953, 52.9260712

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4508549, upper bound: 57.4048137
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4508550, upper bound: 57.4574595
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.4998188, 25.8336639, -7.3768864, 32.1902809, -37.6900902, 33.2105484
1: -7.0067005, 29.4249306, -9.4218798, 36.5786400, -43.5853386, 38.8468094
2: -7.0890341, 28.5652390, -9.3372574, 35.9144287, -43.0034599, 37.9024849
3: -12.4047985, 31.5319157, -16.1180038, 39.1414795, -51.5462761, 47.6499176
4: -11.8526163, 29.3220711, -15.1752281, 36.9083405, -48.7609520, 44.4972992

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4001899, upper bound: 57.4416854
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3489096, upper bound: 57.4443857
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8599839, 30.2850361, -7.3768864, 32.1902809, -39.0502548, 37.6619225
1: -8.7821398, 34.4138641, -9.4218798, 36.5786400, -45.3607750, 43.8357430
2: -8.7217379, 33.7218704, -9.3372574, 35.9144287, -44.6361656, 43.0591278
3: -15.1452637, 36.9015083, -16.1180038, 39.1414795, -54.2867432, 53.0195122
4: -14.1925945, 34.7436333, -15.1752281, 36.9083405, -51.1009331, 49.9188614

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4001899, upper bound: 57.4867045
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3489096, upper bound: 57.4719537
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -7.7393336, 33.4550667, -7.8175883, 33.8507843, -41.5901184, 41.2726555
1: -9.9046087, 37.9284363, -9.9857235, 38.3829613, -48.2875710, 47.9141617
2: -9.7569246, 37.4305687, -9.8629465, 37.8730621, -47.6299820, 47.2935104
3: -17.0155411, 40.4901886, -17.0806026, 41.0101547, -58.0256958, 57.5707855
4: -15.7923479, 38.5961914, -15.8571148, 38.9807281, -54.7730713, 54.4533043

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4377456
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4583964
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -7.7393336, 33.4550667, -8.1414795, 34.9803314, -42.7196655, 41.5965462
1: -9.9046087, 37.9284363, -10.3911715, 39.6709862, -49.5755959, 48.3196068
2: -9.7569246, 37.4305687, -10.2646151, 39.1024780, -48.8593979, 47.6951790
3: -17.0155411, 40.4901886, -17.6921673, 42.4123306, -59.4278679, 58.1823578
4: -15.7923479, 38.5961914, -16.5152435, 40.2248878, -56.0172310, 55.1114349

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4842245
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3922841, upper bound: 57.5005542
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -5.9096947, 27.3499146, -8.0627384, 35.7259407, -41.6356354, 35.4126511
1: -7.5164185, 31.1457329, -10.3429604, 40.4529800, -47.9693985, 41.4886894
2: -7.6001306, 30.2378502, -10.1579933, 39.9914627, -47.5915947, 40.3958435
3: -13.2140646, 33.4282455, -17.8792992, 43.0726242, -56.2866859, 51.3075447
4: -12.7368975, 30.9877415, -16.5315685, 41.0846443, -53.8215408, 47.5193062

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3349874, upper bound: 57.2397932
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4066139, upper bound: 57.3889564
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3467827, 32.0613708, -8.0627384, 35.7259407, -43.0727196, 40.1241074
1: -9.3942413, 36.4329567, -10.3429604, 40.4529800, -49.8472176, 46.7759132
2: -9.3245058, 35.7049446, -10.1579933, 39.9914627, -49.3159676, 45.8629303
3: -16.1041088, 39.0988007, -17.8792992, 43.0726242, -59.1767273, 56.9780998
4: -15.1873417, 36.7418594, -16.5315685, 41.0846443, -56.2719879, 53.2734222

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2960164, upper bound: 57.3687815
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4082406
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -5.9096947, 27.3499146, -7.4656463, 33.3557091, -39.2654037, 34.8155518
1: -7.5164185, 31.1457329, -9.5422745, 37.8093300, -45.3257484, 40.6880035
2: -7.6001306, 30.2378502, -9.4745121, 37.1891479, -44.7892761, 39.7123604
3: -13.2140646, 33.4282455, -16.5416107, 40.3914528, -53.6055183, 49.9698563
4: -12.7368975, 30.9877415, -15.5500689, 38.1762428, -50.9131393, 46.5378075

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4087402, upper bound: 57.3565637
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4807124, upper bound: 57.4807124
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -7.3467827, 32.0613708, -7.4656463, 33.3557091, -40.7024841, 39.5270157
1: -9.3942413, 36.4329567, -9.5422745, 37.8093300, -47.2035713, 45.9752274
2: -9.3245058, 35.7049446, -9.4745121, 37.1891479, -46.5136528, 45.1794510
3: -16.1041088, 39.0988007, -16.5416107, 40.3914528, -56.4955597, 55.6404037
4: -15.1873417, 36.7418594, -15.5500689, 38.1762428, -53.3635864, 52.2919273

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4616763, upper bound: 57.4996189
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4814832, upper bound: 57.4995890
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -6.9876323, 30.5188522, -8.6143875, 36.8573952, -43.8450279, 39.1332397
1: -8.9284620, 34.6611366, -11.0205164, 41.7574120, -50.6858749, 45.6816406
2: -8.8685627, 33.9844780, -10.7981386, 41.3458519, -50.2144089, 44.7826157
3: -15.3430519, 37.1408958, -18.7542477, 44.5086746, -59.8517265, 55.8951416
4: -14.4918509, 34.9908066, -17.2629986, 42.5251923, -57.0170441, 52.2538071

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4541660, upper bound: 57.4472563
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4335301, upper bound: 57.4194369
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -7.1687813, 31.2684269, -8.8100138, 37.5939674, -44.7627487, 40.0784378
1: -9.1591225, 35.5011978, -11.2688313, 42.5857544, -51.7448654, 46.7700272
2: -9.0866833, 34.8562317, -11.0355730, 42.1981087, -51.2847900, 45.8918037
3: -15.7503662, 38.0228539, -19.1602993, 45.3824959, -61.1328621, 57.1831512
4: -14.8445187, 35.8955460, -17.6283436, 43.3953476, -58.2398682, 53.5238762

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4043083, upper bound: 57.4375265
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4043080, upper bound: 57.4573161
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -6.9876323, 30.5188522, -8.5083008, 36.3890610, -43.3766937, 39.0271530
1: -8.9284620, 34.6611366, -10.8575277, 41.2442513, -50.1727142, 45.5186577
2: -8.8685627, 33.9844780, -10.7077875, 40.7232666, -49.5918236, 44.6922646
3: -15.3430519, 37.1408958, -18.4530334, 44.0617104, -59.4047623, 55.5939293
4: -14.4918509, 34.9908066, -17.1977558, 41.8714981, -56.3633461, 52.1885605

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4937070, upper bound: 57.5098514
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4837925, upper bound: 57.4856520
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -7.1687813, 31.2684269, -8.7081556, 37.1425133, -44.3112946, 39.9765816
1: -9.1591225, 35.5011978, -11.1119413, 42.0899849, -51.2490997, 46.6131401
2: -9.0866833, 34.8562317, -10.9512491, 41.5937920, -50.6804733, 45.8074799
3: -15.7503662, 38.0228539, -18.8729000, 44.9509163, -60.7012825, 56.8957520
4: -14.8445187, 35.8955460, -17.5746422, 42.7677650, -57.6122818, 53.4701843

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4795312, upper bound: 57.5126542
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4795312, upper bound: 57.5306095
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.6970072, 34.3913345, -6.2563448, 28.6933155, -36.3903198, 40.6476784
1: -9.8640242, 38.9474487, -7.9915113, 32.5977592, -42.4617767, 46.9389534
2: -9.7420282, 38.4363785, -7.9890079, 31.8166008, -41.5586205, 46.4253769
3: -17.1286697, 41.5261650, -14.0460072, 34.9063377, -52.0349998, 55.5721741
4: -15.9208412, 39.5123138, -13.2832928, 32.6852341, -48.6060753, 52.7955971

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3643755, upper bound: 57.3447240
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3378868, upper bound: 57.3327170
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.4656463, 33.3557091, -6.2563448, 28.6933155, -36.1589584, 39.6120529
1: -9.5422745, 37.8093300, -7.9915113, 32.5977592, -42.1400223, 45.8008423
2: -9.4745121, 37.1891479, -7.9890079, 31.8166008, -41.2911034, 45.1781540
3: -16.5416107, 40.3914528, -14.0460072, 34.9063377, -51.4479485, 54.4374619
4: -15.5500689, 38.1762428, -13.2832928, 32.6852341, -48.2353020, 51.4595337

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3536767, upper bound: 57.4112061
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3378868, upper bound: 57.3533563
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -7.2943954, 32.5972137, -6.5591278, 30.0860119, -37.3804092, 39.1563377
1: -9.3045883, 36.9583969, -8.4003382, 34.2025681, -43.5071564, 45.3587341
2: -9.2544870, 36.2896118, -8.3580723, 33.4477882, -42.7022743, 44.6476822
3: -16.1635914, 39.5124702, -14.7368250, 36.5724945, -52.7360802, 54.2492943
4: -15.2678089, 37.2344894, -13.8810635, 34.3726807, -49.6404800, 51.1155510

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3657514, upper bound: 57.4244031
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3657514, upper bound: 57.4741026
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.3103456, 36.4285011, -7.0722103, 31.2705612, -39.5809021, 43.5007095
1: -10.6107883, 41.2406464, -9.0512457, 35.5169983, -46.1277809, 50.2918930
2: -10.4886055, 40.7226562, -8.9803925, 34.8309250, -45.3195229, 49.7030487
3: -18.2756023, 43.9959564, -15.6104374, 38.0437126, -56.3193092, 59.6063881
4: -17.0999012, 41.8161430, -14.6354523, 35.8261719, -52.9260712, 56.4515953

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4048137, upper bound: 57.4508550
time: 0.61 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4048137, upper bound: 57.4771421
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3768864, 32.1902809, -5.4998188, 25.8336639, -33.2105484, 37.6900902
1: -9.4218798, 36.5786400, -7.0067005, 29.4249306, -38.8468094, 43.5853386
2: -9.3372574, 35.9144287, -7.0890341, 28.5652390, -37.9024849, 43.0034599
3: -16.1180038, 39.1414795, -12.4047985, 31.5319157, -47.6499176, 51.5462723
4: -15.1752281, 36.9083405, -11.8526163, 29.3220711, -44.4972992, 48.7609520

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4416853, upper bound: 57.4001899
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4443857, upper bound: 57.3720362
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3768864, 32.1902809, -6.8599839, 30.2850361, -37.6619225, 39.0502548
1: -9.4218798, 36.5786400, -8.7821398, 34.4138641, -43.8357430, 45.3607788
2: -9.3372574, 35.9144287, -8.7217379, 33.7218704, -43.0591278, 44.6361656
3: -16.1180038, 39.1414795, -15.1452637, 36.9015083, -53.0195122, 54.2867432
4: -15.1752281, 36.9083405, -14.1925945, 34.7436333, -49.9188614, 51.1009331

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4416854, upper bound: 57.4811964
time: 0.57 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4443858, upper bound: 57.4532558
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -7.8175883, 33.8507843, -7.7393336, 33.4550667, -41.2726555, 41.5901184
1: -9.9857235, 38.3829613, -9.9046087, 37.9284363, -47.9141617, 48.2875710
2: -9.8629465, 37.8730621, -9.7569246, 37.4305687, -47.2935143, 47.6299820
3: -17.0806026, 41.0101547, -17.0155411, 40.4901886, -57.5707893, 58.0256958
4: -15.8571148, 38.9807281, -15.7923479, 38.5961914, -54.4533043, 54.7730713

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4377456, upper bound: 57.4036201
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4377457, upper bound: 57.4629417
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -8.1414795, 34.9803314, -7.7393336, 33.4550667, -41.5965462, 42.7196655
1: -10.3911715, 39.6709862, -9.9046087, 37.9284363, -48.3196068, 49.5755959
2: -10.2646151, 39.1024780, -9.7569246, 37.4305687, -47.6951790, 48.8593979
3: -17.6921673, 42.4123306, -17.0155411, 40.4901886, -58.1823578, 59.4278679
4: -16.5152435, 40.2248878, -15.7923479, 38.5961914, -55.1114349, 56.0172310

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4377456, upper bound: 57.4304013
time: 0.61 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4377457, upper bound: 57.4629417
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -5.9096947, 27.3499146, -35.4126511, 41.6356354
1: -10.3429604, 40.4529800, -7.5164185, 31.1457329, -41.4886894, 47.9693985
2: -10.1579933, 39.9914627, -7.6001306, 30.2378502, -40.3958435, 47.5915909
3: -17.8792992, 43.0726242, -13.2140646, 33.4282455, -51.3075447, 56.2866859
4: -16.5315685, 41.0846443, -12.7368975, 30.9877415, -47.5193062, 53.8215408

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2397932, upper bound: 57.3349874
time: 0.51 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3889564, upper bound: 57.4066139
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -7.3467827, 32.0613708, -40.1241074, 43.0727196
1: -10.3429604, 40.4529800, -9.3942413, 36.4329567, -46.7759132, 49.8472176
2: -10.1579933, 39.9914627, -9.3245058, 35.7049446, -45.8629303, 49.3159676
3: -17.8792992, 43.0726242, -16.1041088, 39.0988007, -56.9780998, 59.1767273
4: -16.5315685, 41.0846443, -15.1873417, 36.7418594, -53.2734222, 56.2719879

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2542380, upper bound: 57.4467256
time: 0.61 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4593619
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7.4656463, 33.3557091, -5.9096947, 27.3499146, -34.8155556, 39.2654037
1: -9.5422745, 37.8093300, -7.5164185, 31.1457329, -40.6880035, 45.3257484
2: -9.4745121, 37.1891479, -7.6001306, 30.2378502, -39.7123604, 44.7892799
3: -16.5416107, 40.3914528, -13.2140646, 33.4282455, -49.9698563, 53.6055183
4: -15.5500689, 38.1762428, -12.7368975, 30.9877415, -46.5378113, 50.9131393

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4564147, upper bound: 57.4510560
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4949370, upper bound: 57.4859599
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7.4656463, 33.3557091, -7.3467827, 32.0613708, -39.5270157, 40.7024879
1: -9.5422745, 37.8093300, -9.3942413, 36.4329567, -45.9752235, 47.2035713
2: -9.4745121, 37.1891479, -9.3245058, 35.7049446, -45.1794510, 46.5136528
3: -16.5416107, 40.3914528, -16.1041088, 39.0988007, -55.6404037, 56.4955597
4: -15.5500689, 38.1762428, -15.1873417, 36.7418594, -52.2919273, 53.3635864

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4836270, upper bound: 57.5228231
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4912707, upper bound: 57.5228231
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6143875, 36.8573952, -6.9876323, 30.5188522, -39.1332397, 43.8450279
1: -11.0205164, 41.7574120, -8.9284620, 34.6611366, -45.6816406, 50.6858749
2: -10.7981386, 41.3458519, -8.8685627, 33.9844780, -44.7826157, 50.2144089
3: -18.7542477, 44.5086746, -15.3430519, 37.1408958, -55.8951416, 59.8517265
4: -17.2629986, 42.5251923, -14.4918509, 34.9908066, -52.2538071, 57.0170441

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4472563, upper bound: 57.4541660
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4194369, upper bound: 57.4480721
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -8.8100138, 37.5939674, -7.1687813, 31.2684269, -40.0784416, 44.7627487
1: -11.2688313, 42.5857544, -9.1591225, 35.5011978, -46.7700272, 51.7448654
2: -11.0355730, 42.1981087, -9.0866833, 34.8562317, -45.8918037, 51.2847900
3: -19.1602993, 45.3824959, -15.7503662, 38.0228539, -57.1831512, 61.1328621
4: -17.6283436, 43.3953476, -14.8445187, 35.8955460, -53.5238762, 58.2398682

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4375265, upper bound: 57.4098177
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4375266, upper bound: 57.4659895
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -8.5083008, 36.3890610, -6.9876323, 30.5188522, -39.0271530, 43.3766937
1: -10.8575277, 41.2442513, -8.9284620, 34.6611366, -45.5186577, 50.1727142
2: -10.7077875, 40.7232666, -8.8685627, 33.9844780, -44.6922646, 49.5918236
3: -18.4530334, 44.0617104, -15.3430519, 37.1408958, -55.5939293, 59.4047623
4: -17.1977558, 41.8714981, -14.4918509, 34.9908066, -52.1885605, 56.3633499

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5146295, upper bound: 57.4953659
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4925580, upper bound: 57.4876294
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -8.7081556, 37.1425133, -7.1687813, 31.2684269, -39.9765816, 44.3112946
1: -11.1119413, 42.0899849, -9.1591225, 35.5011978, -46.6131401, 51.2490997
2: -10.9512491, 41.5937920, -9.0866833, 34.8562317, -45.8074799, 50.6804733
3: -18.8729000, 44.9509163, -15.7503662, 38.0228539, -56.8957520, 60.7012825
4: -17.5746422, 42.7677650, -14.8445187, 35.8955460, -53.4701881, 57.6122818

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5216208, upper bound: 57.4937772
time: 0.60 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5216208, upper bound: 57.5321283
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -8.0627384, 35.7259407, -43.7886810, 43.7886810
1: -10.3429604, 40.4529800, -10.3429604, 40.4529800, -50.7959366, 50.7959366
2: -10.1579933, 39.9914627, -10.1579933, 39.9914627, -50.1494522, 50.1494522
3: -17.8792992, 43.0726242, -17.8792992, 43.0726242, -60.9519234, 60.9519234
4: -16.5315685, 41.0846443, -16.5315685, 41.0846443, -57.6162109, 57.6162071

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3502461, upper bound: 57.3646597
time: 0.59 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3327170, upper bound: 57.3347858
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -9.0433788, 38.5837364, -8.0627384, 35.7259407, -44.7693176, 46.6464729
1: -11.5689020, 43.6965904, -10.3429604, 40.4529800, -52.0218811, 54.0395508
2: -11.3229685, 43.3287659, -10.1579933, 39.9914627, -51.3144302, 53.4867592
3: -19.6716652, 46.5399094, -17.8792992, 43.0726242, -62.7442856, 64.4192047
4: -18.0751343, 44.5710564, -16.5315685, 41.0846443, -59.1597786, 61.1026115

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3643755, upper bound: 57.3623975
time: 0.60 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3378641, upper bound: 57.3558346
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -7.4656463, 33.3557091, -8.0627384, 35.7259407, -43.1915855, 41.4184494
1: -9.5422745, 37.8093300, -10.3429604, 40.4529800, -49.9952545, 48.1522903
2: -9.4745121, 37.1891479, -10.1579933, 39.9914627, -49.4659729, 47.3471413
3: -16.5416107, 40.3914528, -17.8792992, 43.0726242, -59.6142311, 58.2707520
4: -15.5500689, 38.1762428, -16.5315685, 41.0846443, -56.6347122, 54.7078094

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4111816, upper bound: 57.3897373
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3860516, upper bound: 57.3564541
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -8.7081556, 37.1425133, -8.0627384, 35.7259407, -44.4340973, 45.2052536
1: -11.1119413, 42.0899849, -10.3429604, 40.4529800, -51.5649223, 52.4329453
2: -10.9512491, 41.5937920, -10.1579933, 39.9914627, -50.9427109, 51.7517815
3: -18.8729000, 44.9509163, -17.8792992, 43.0726242, -61.9455261, 62.8302155
4: -17.5746422, 42.7677650, -16.5315685, 41.0846443, -58.6592865, 59.2993279

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3842770, upper bound: 57.3920057
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4229728, upper bound: 57.4107255
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -7.6815906, 34.2567787, -9.0433788, 38.5837364, -46.2653275, 43.3001556
1: -9.8552742, 38.7963371, -11.5689020, 43.6965904, -53.5518646, 50.3652382
2: -9.7000809, 38.3169556, -11.3229685, 43.3287659, -53.0288467, 49.6399231
3: -17.0625191, 41.3339920, -19.6716652, 46.5399094, -63.6024284, 61.0056572
4: -15.8016033, 39.3707428, -18.0751343, 44.5710564, -60.3726463, 57.4458771

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3502461, upper bound: 57.4200947
time: 0.57 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3378641, upper bound: 57.4083992
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -8.8967991, 37.9971581, -9.0433788, 38.5837364, -47.4805374, 47.0405350
1: -11.3845472, 43.0349770, -11.5689020, 43.6965904, -55.0811348, 54.6038780
2: -11.1436529, 42.6596985, -11.3229685, 43.3287659, -54.4724197, 53.9826660
3: -19.3649063, 45.8372116, -19.6716652, 46.5399094, -65.9048157, 65.5088806
4: -17.7870979, 43.8872910, -18.0751343, 44.5710564, -62.3581390, 61.9624252

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3585705, upper bound: 57.4078675
time: 0.57 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3618004, upper bound: 57.3987345
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -7.0637846, 31.8527794, -9.0433788, 38.5837364, -45.6475182, 40.8961563
1: -9.0256519, 36.1201248, -11.5689020, 43.6965904, -52.7222443, 47.6890259
2: -8.9952803, 35.4699478, -11.3229685, 43.3287659, -52.3240433, 46.7929115
3: -15.6883984, 38.6212540, -19.6716652, 46.5399094, -62.2283096, 58.2929153
4: -14.7898903, 36.4125786, -18.0751343, 44.5710564, -59.3609428, 54.4877129

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4009100, upper bound: 57.4385660
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4229728, upper bound: 57.4477043
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.6094341, 36.7576447, -9.0433788, 38.5837364, -47.1931686, 45.8010254
1: -10.9853382, 41.6540833, -11.5689020, 43.6965904, -54.6819305, 53.2229843
2: -10.8296928, 41.1556931, -11.3229685, 43.3287659, -54.1584587, 52.4786606
3: -18.6601658, 44.4872971, -19.6716652, 46.5399094, -65.2000732, 64.1589661
4: -17.3866043, 42.3157730, -18.0751343, 44.5710564, -61.9576607, 60.3909035

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4009100, upper bound: 57.4559145
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4229728, upper bound: 57.4639132
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -7.4656463, 33.3557091, -41.4184494, 43.1915855
1: -10.3429604, 40.4529800, -9.5422745, 37.8093300, -48.1522903, 49.9952545
2: -10.1579933, 39.9914627, -9.4745121, 37.1891479, -47.3471413, 49.4659729
3: -17.8792992, 43.0726242, -16.5416107, 40.3914528, -58.2707520, 59.6142311
4: -16.5315685, 41.0846443, -15.5500689, 38.1762428, -54.7078094, 56.6347122

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3719413, upper bound: 57.3676960
time: 0.55 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3493657, upper bound: 57.3617395
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -8.0627384, 35.7259407, -8.7081556, 37.1425133, -45.2052536, 44.4340973
1: -10.3429604, 40.4529800, -11.1119413, 42.0899849, -52.4329453, 51.5649147
2: -10.1579933, 39.9914627, -10.9512491, 41.5937920, -51.7517815, 50.9427109
3: -17.8792992, 43.0726242, -18.8729000, 44.9509163, -62.8302116, 61.9455261
4: -16.5315685, 41.0846443, -17.5746422, 42.7677650, -59.2993279, 58.6592865

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2542380, upper bound: 57.4467256
time: 0.50 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3896511, upper bound: 57.4593619
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -9.0433788, 38.5837364, -7.0637846, 31.8527794, -40.8961563, 45.6475220
1: -11.5689020, 43.6965904, -9.0256519, 36.1201248, -47.6890259, 52.7222443
2: -11.3229685, 43.3287659, -8.9952803, 35.4699478, -46.7929115, 52.3240433
3: -19.6716652, 46.5399094, -15.6883984, 38.6212540, -58.2929153, 62.2283096
4: -18.0751343, 44.5710564, -14.7898903, 36.4125786, -54.4877129, 59.3609428

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4255378, upper bound: 57.3750661
time: 0.61 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4120202, upper bound: 57.3695380
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -9.0433788, 38.5837364, -8.6094341, 36.7576447, -45.8010254, 47.1931686
1: -11.5689020, 43.6965904, -10.9853382, 41.6540833, -53.2229843, 54.6819305
2: -11.3229685, 43.3287659, -10.8296928, 41.1556931, -52.4786606, 54.1584587
3: -19.6716652, 46.5399094, -18.6601658, 44.4872971, -64.1589661, 65.2000732
4: -18.0751343, 44.5710564, -17.3866043, 42.3157730, -60.3909035, 61.9576530

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4271527, upper bound: 57.4199984
time: 0.62 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4004376, upper bound: 57.4144006
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -6.0869403, 28.1034451, -9.8139334, 41.2615242, -47.3484650, 37.9173775
1: -7.7611790, 31.9777794, -12.5014868, 46.7032166, -54.4643898, 44.4792671
2: -7.8130293, 31.1800365, -12.2750893, 46.4147034, -54.2277336, 43.4551201
3: -13.6107702, 34.2371445, -21.2191811, 49.7177658, -63.3285370, 55.4563141
4: -12.9863300, 32.0082283, -19.6729584, 47.8018913, -60.7882118, 51.6811867

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A2_A1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1649963, upper bound: 57.2608814
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5151167, upper bound: 57.5277506
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -6.8298445, 30.9102707, -10.2187920, 42.7633629, -49.5931969, 41.1290588
1: -8.7239046, 35.0731049, -13.0090141, 48.3895683, -57.1134682, 48.0821190
2: -8.7083988, 34.3763885, -12.7634277, 48.1404190, -56.8488159, 47.1398125
3: -15.1984425, 37.5194168, -22.0383968, 51.5057220, -66.7041626, 59.5578156
4: -14.3523102, 35.2999115, -20.4505177, 49.5547600, -63.9070511, 55.7504196

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5048539, upper bound: 57.5244702
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5048539, upper bound: 57.5244702
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -6.6227660, 29.5530052, -8.5659733, 36.5436745, -43.1664391, 38.1189804
1: -8.4487925, 33.6364212, -10.9255056, 41.3838005, -49.8325920, 44.5619240
2: -8.4402018, 32.8575706, -10.7740145, 40.9791374, -49.4193382, 43.6315842
3: -14.5583429, 36.0417862, -18.6278038, 44.1337662, -58.6921082, 54.6695900
4: -13.8166447, 33.7440796, -17.3158875, 42.2269249, -56.0435677, 51.0599632

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4645058, upper bound: 57.4668672
time: 0.51 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5381886, upper bound: 57.5361746
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -8.1414795, 34.9803314, -8.9339600, 37.9039001, -46.0453758, 43.9142914
1: -10.3911715, 39.6709862, -11.3865700, 42.9051590, -53.2963295, 51.0575562
2: -10.2646151, 39.1024780, -11.2176380, 42.5449905, -52.8095932, 50.3201141
3: -17.6921673, 42.4123306, -19.3760033, 45.7466164, -63.4387703, 61.7883301
4: -16.5152435, 40.2248878, -18.0218716, 43.8178978, -60.3331375, 58.2467537

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5348344, upper bound: 57.5348344
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5348344, upper bound: 57.5348344
time: 1.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.54 seconds
NS_A1_B1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3535846, upper bound: 57.3859768
NS_A1_B1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3333625, upper bound: 57.3333625
NS_A1_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3859768, upper bound: 57.4075307
NS_A1_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3333625, upper bound: 57.3779775
NS_A1_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4031152, upper bound: 57.4704486
NS_A1_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4426372, upper bound: 57.4955818
NS_A1_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4859338
NS_A1_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4979221
NS_A1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4540048, upper bound: 57.3776547
NS_A1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4113590, upper bound: 57.3538622
NS_A1_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4019974, upper bound: 57.4651135
NS_A1_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4113592, upper bound: 57.4465134
NS_A1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4807931, upper bound: 57.4787224
NS_A1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4080314, upper bound: 57.4485118
NS_A1_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4910985, upper bound: 57.4707716
NS_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4910986, upper bound: 57.5146976
NS_A1_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4075307, upper bound: 57.4022774
NS_A1_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3779775, upper bound: 57.3513281
NS_A1_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4354261, upper bound: 57.4446294
NS_A1_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4521395
NS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4787224, upper bound: 57.4807931
NS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4485118, upper bound: 57.4080314
NS_A1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4494365, upper bound: 57.4910985
NS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4494365, upper bound: 57.5073460
NS_A1_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4557617, upper bound: 57.4282429
NS_A1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4164057, upper bound: 57.4164057
NS_A1_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4354261, upper bound: 57.4945006
NS_A1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4515008, upper bound: 57.4945006
NS_A1_B1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5151404, upper bound: 57.5111715
NS_A1_B1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5315038, upper bound: 57.5316335
NS_A1_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
NS_A1_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
NS_A1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3447240, upper bound: 57.3643755
NS_A1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3327170, upper bound: 57.3378868
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3812309, upper bound: 57.4167830
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3327170, upper bound: 57.3838189
NS_A1_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4244031, upper bound: 57.3657514
NS_A1_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4244031, upper bound: 57.4385264
NS_A1_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4508549, upper bound: 57.4048137
NS_A1_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4508550, upper bound: 57.4574595
NS_A1_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4001899, upper bound: 57.4416854
NS_A1_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3489096, upper bound: 57.4443857
NS_A1_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4001899, upper bound: 57.4867045
NS_A1_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3489096, upper bound: 57.4719537
NS_A1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4377456
NS_A1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4583964
NS_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3922841, upper bound: 57.4842245
NS_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3922841, upper bound: 57.5005542
NS_A1_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3349874, upper bound: 57.2397932
NS_A1_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4066139, upper bound: 57.3889564
NS_A1_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.2960164, upper bound: 57.3687815
NS_A1_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4072148, upper bound: 57.4082406
NS_A1_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4087402, upper bound: 57.3565637
NS_A1_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4807124, upper bound: 57.4807124
NS_A1_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4616763, upper bound: 57.4996189
NS_A1_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4814832, upper bound: 57.4995890
NS_A1_B2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4541660, upper bound: 57.4472563
NS_A1_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4335301, upper bound: 57.4194369
NS_A1_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4043083, upper bound: 57.4375265
NS_A1_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4043080, upper bound: 57.4573161
NS_A1_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4937070, upper bound: 57.5098514
NS_A1_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4837925, upper bound: 57.4856520
NS_A1_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4795312, upper bound: 57.5126542
NS_A1_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4795312, upper bound: 57.5306095
NS_A2_B1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3643755, upper bound: 57.3447240
NS_A2_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3378868, upper bound: 57.3327170
NS_A2_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3536767, upper bound: 57.4112061
NS_A2_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3378868, upper bound: 57.3533563
NS_A2_B1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3657514, upper bound: 57.4244031
NS_A2_B1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3657514, upper bound: 57.4741026
NS_A2_B1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4048137, upper bound: 57.4508550
NS_A2_B1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4048137, upper bound: 57.4771421
NS_A2_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4416853, upper bound: 57.4001899
NS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4443857, upper bound: 57.3720362
NS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4416854, upper bound: 57.4811964
NS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4443858, upper bound: 57.4532558
NS_A2_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4377456, upper bound: 57.4036201
NS_A2_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4377457, upper bound: 57.4629417
NS_A2_B1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4377456, upper bound: 57.4304013
NS_A2_B1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4377457, upper bound: 57.4629417
NS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.2397932, upper bound: 57.3349874
NS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3889564, upper bound: 57.4066139
NS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.2542380, upper bound: 57.4467256
NS_A2_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3852977, upper bound: 57.4593619
NS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4564147, upper bound: 57.4510560
NS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4949370, upper bound: 57.4859599
NS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4836270, upper bound: 57.5228231
NS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4912707, upper bound: 57.5228231
NS_A2_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4472563, upper bound: 57.4541660
NS_A2_B1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4194369, upper bound: 57.4480721
NS_A2_B1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4375265, upper bound: 57.4098177
NS_A2_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4375266, upper bound: 57.4659895
NS_A2_B1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5146295, upper bound: 57.4953659
NS_A2_B1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4925580, upper bound: 57.4876294
NS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5216208, upper bound: 57.4937772
NS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5216208, upper bound: 57.5321283
NS_A2_B2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3502461, upper bound: 57.3646597
NS_A2_B2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3327170, upper bound: 57.3347858
NS_A2_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3643755, upper bound: 57.3623975
NS_A2_B2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3378641, upper bound: 57.3558346
NS_A2_B2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4111816, upper bound: 57.3897373
NS_A2_B2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3860516, upper bound: 57.3564541
NS_A2_B2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3842770, upper bound: 57.3920057
NS_A2_B2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4229728, upper bound: 57.4107255
NS_A2_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3502461, upper bound: 57.4200947
NS_A2_B2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3378641, upper bound: 57.4083992
NS_A2_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3585705, upper bound: 57.4078675
NS_A2_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3618004, upper bound: 57.3987345
NS_A2_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4009100, upper bound: 57.4385660
NS_A2_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4229728, upper bound: 57.4477043
NS_A2_B2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4009100, upper bound: 57.4559145
NS_A2_B2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4229728, upper bound: 57.4639132
NS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3719413, upper bound: 57.3676960
NS_A2_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3493657, upper bound: 57.3617395
NS_A2_B2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.2542380, upper bound: 57.4467256
NS_A2_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.3896511, upper bound: 57.4593619
NS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4255378, upper bound: 57.3750661
NS_A2_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4120202, upper bound: 57.3695380
NS_A2_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4271527, upper bound: 57.4199984
NS_A2_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4004376, upper bound: 57.4144006
NS_A2_B2_B2_A2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.1649963, upper bound: 57.2608814
NS_A2_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5151167, upper bound: 57.5277506
NS_A2_B2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5048539, upper bound: 57.5244702
NS_A2_B2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5048539, upper bound: 57.5244702
NS_A2_B2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.4645058, upper bound: 57.4668672
NS_A2_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5381886, upper bound: 57.5361746
NS_A2_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5348344, upper bound: 57.5348344
NS_A2_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -57.5348344, upper bound: 57.5348344

## BFS NS instance: NS_A1_B1_A1_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -5.8440404, 27.0814819, -5.2112594, 24.7932606, -30.6373005, 32.2927399
1: -7.4554582, 30.8000145, -6.6182418, 28.2828865, -35.7383461, 37.4182549
2: -7.4994626, 29.9851818, -6.7461748, 27.3695774, -34.8690376, 36.7313499
3: -13.1411705, 33.0097122, -11.7350798, 30.3101082, -43.4512749, 44.7447815
4: -12.4984608, 30.7952538, -11.3421888, 28.0153084, -40.5137711, 42.1374435

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3239815, upper bound: 57.3494894
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3485427, upper bound: 57.3823323
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -6.2563448, 28.6933155, -5.5381598, 26.1057167, -32.3620605, 34.2314682
1: -7.9915113, 32.5977592, -7.0475054, 29.7262764, -37.7177849, 39.6452560
2: -7.9890079, 31.8166008, -7.1299081, 28.8404236, -36.8294220, 38.9465065
3: -14.0460072, 34.9063377, -12.4831791, 31.8344269, -45.8804321, 47.3895187
4: -13.2832928, 32.6852341, -11.9530659, 29.5439148, -42.8272018, 44.6382980

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3333625, upper bound: 57.3333625
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3333625, upper bound: 57.3333625
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -5.2112594, 24.7932606, -5.4860497, 25.6992893, -30.9105415, 30.2793102
1: -6.6182418, 28.2828865, -6.9626961, 29.3125572, -35.9307976, 35.2455826
2: -6.7461748, 27.3695774, -7.0903730, 28.3714809, -35.1176491, 34.4599495
3: -11.7350798, 30.3101082, -12.2716236, 31.4735260, -43.2086029, 42.5817337
4: -11.3421888, 28.0153084, -11.9300127, 29.0408840, -40.3830719, 39.9453201

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3639092, upper bound: 57.3697377
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4091652, upper bound: 57.4033179
time: 0.55 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.24 + 417.23 = 420.48 seconds
