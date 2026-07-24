## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 146.59001129824


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348)
1: (-57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504)
2: (-47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090)
3: (-74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187)
4: (-62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 2.09 = 3.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -146.7073772, upper bound: 146.7073772

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7005928, upper bound: 146.7069990
time: 0.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.37 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 0, lower bound: -146.7005928, upper bound: 146.7069990
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -57.5744476, 75.5054245, -67.8345108, 89.3525009, -146.9269409, 143.3398895
1: -45.2290192, 62.4155273, -53.8048248, 73.8109055, -119.0399094, 116.2203369
2: -37.4250565, 62.9155426, -44.4119987, 74.3603668, -111.7854233, 107.3275299
3: -59.0007706, 75.3333282, -70.0837402, 89.2093430, -148.2101135, 145.4170685
4: -49.6813240, 83.8791275, -58.8266220, 99.3475266, -149.0288544, 142.7057495

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -63.5754318, 83.6630020, -70.2021866, 92.7214432, -156.2968597, 153.8651886
1: -50.3897362, 68.7696991, -55.8390121, 76.5503235, -126.9400635, 124.6087112
2: -41.4750328, 69.6277542, -46.0420609, 77.1488266, -118.6238403, 115.6698151
3: -65.6504517, 83.2662277, -72.7555389, 92.5662308, -158.2166443, 156.0217590
4: -54.9593964, 92.7353210, -60.9916496, 103.1074448, -158.0668335, 153.7269592

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.87 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.96 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -57.5744476, 75.5054245, -57.5744476, 75.5054245, -133.0798340, 133.0798492
1: -45.2290192, 62.4155273, -45.2290192, 62.4155273, -107.6445236, 107.6445236
2: -37.4250565, 62.9155426, -37.4250565, 62.9155426, -100.3405914, 100.3405914
3: -59.0007706, 75.3333282, -59.0007706, 75.3333282, -134.3341064, 134.3341064
4: -49.6813240, 83.8791275, -49.6813240, 83.8791275, -133.5604401, 133.5604553

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7053808
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7057147
time: 1.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -57.5744476, 75.5054245, -63.5754318, 83.6630020, -141.2374573, 139.0808258
1: -45.2290192, 62.4155273, -50.3897362, 68.7696991, -113.9987106, 112.8052597
2: -37.4250565, 62.9155426, -41.4750328, 69.6277542, -107.0528107, 104.3905563
3: -59.0007706, 75.3333282, -65.6504517, 83.2662277, -142.2669983, 140.9837646
4: -49.6813240, 83.8791275, -54.9593964, 92.7353210, -142.4166412, 138.8385162

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7053808
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7057147
time: 0.74 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -63.5754318, 83.6630020, -57.5744476, 75.5054245, -139.0808258, 141.2374573
1: -50.3897362, 68.7696991, -45.2290192, 62.4155273, -112.8052597, 113.9987106
2: -41.4750328, 69.6277542, -37.4250565, 62.9155426, -104.3905563, 107.0528107
3: -65.6504517, 83.2662277, -59.0007706, 75.3333282, -140.9837494, 142.2669983
4: -54.9593964, 92.7353210, -49.6813240, 83.8791275, -138.8385315, 142.4166412

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7000438, upper bound: 146.6989132
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.62 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -63.5754318, 83.6630020, -63.5754318, 83.6630020, -147.2384338, 147.2384338
1: -50.3897362, 68.7696991, -50.3897362, 68.7696991, -119.1594391, 119.1594391
2: -41.4750328, 69.6277542, -41.4750328, 69.6277542, -111.1027679, 111.1027679
3: -65.6504517, 83.2662277, -65.6504517, 83.2662277, -148.9166565, 148.9166565
4: -54.9593964, 92.7353210, -54.9593964, 92.7353210, -147.6947174, 147.6947174

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7000438, upper bound: 146.6989132
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.93 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7053808
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7057147
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7053808
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.6976578, upper bound: 146.7057147
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.7000438, upper bound: 146.6989132
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.7000438, upper bound: 146.6989132
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -146.7002436, upper bound: 146.7002436

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -43.4482880, 55.6397705, -52.8609428, 69.0925217, -112.5407944, 108.5007172
1: -33.2318993, 45.8998451, -41.3653069, 57.0941544, -90.3260498, 87.2651291
2: -27.5560265, 46.3096161, -34.2345276, 57.5711327, -85.1271439, 80.5441437
3: -43.4949112, 55.4750938, -53.9919853, 68.9537735, -112.4486847, 109.4670715
4: -36.5288200, 61.3431358, -45.4602165, 76.6291656, -113.1579895, 106.8033524

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7060133, upper bound: 146.7059916
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7062746, upper bound: 146.7060347
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -50.0694695, 64.8776169, -56.8402557, 74.4749146, -124.5443802, 121.7178497
1: -38.8252449, 53.5614166, -44.6095352, 61.5551300, -100.3803711, 98.1709518
2: -32.1229935, 54.1428337, -36.9128647, 62.0687027, -94.1916962, 91.0556870
3: -50.5690689, 64.7720032, -58.1832848, 74.2922974, -124.8613586, 122.9552917
4: -42.6325302, 71.8491440, -49.0009956, 82.7187881, -125.3513107, 120.8501434

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061607, upper bound: 146.7063234
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7064280, upper bound: 146.7064280
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -43.4482880, 55.6397705, -58.2211647, 76.4853363, -119.9336243, 113.8609314
1: -33.2318993, 45.8998451, -46.0508804, 62.8897934, -96.1216888, 91.9507217
2: -27.5560265, 46.3096161, -37.9057198, 63.6818886, -91.2379150, 84.2153320
3: -43.4949112, 55.4750938, -60.0653152, 76.1531067, -119.6480179, 115.5404053
4: -36.5288200, 61.3431358, -50.2575073, 84.6902466, -121.2190704, 111.6006470

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7037085
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7053808
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -50.0694695, 64.8776169, -62.9154472, 82.7271881, -132.7966614, 127.7930450
1: -38.8252449, 53.5614166, -49.8315773, 67.9938202, -106.8190613, 103.3929901
2: -32.1229935, 54.1428337, -41.0178070, 68.8530579, -100.9760513, 95.1606293
3: -50.5690689, 64.7720032, -64.9094162, 82.3290024, -132.8980713, 129.6814270
4: -42.6325302, 71.8491440, -54.3509293, 91.6772003, -134.3097229, 126.2000732

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873503, upper bound: 146.7037421
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6977803, upper bound: 146.7057147
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -69.9139099, 92.8095551, -54.6027412, 71.4642410, -141.3781433, 147.4122925
1: -56.2212448, 76.4052200, -42.8464355, 59.0657921, -115.2870255, 119.2516403
2: -46.1604691, 76.7849960, -35.4524841, 59.5200462, -105.6805115, 112.2374802
3: -73.1382370, 92.7751694, -55.9210243, 71.3060074, -144.4442444, 148.6961975
4: -61.1051254, 102.5622787, -47.0517082, 79.3208847, -140.4259949, 149.6139832

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7037085, upper bound: 146.6872421
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7035058, upper bound: 146.6873503
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -60.9523659, 80.0911331, -56.9579620, 74.6815186, -135.6338806, 137.0490723
1: -48.2995453, 65.8306274, -44.7406845, 61.7294083, -110.0289307, 110.5713120
2: -39.7593002, 66.6049957, -37.0215836, 62.2200584, -101.9793472, 103.6265717
3: -62.9388199, 79.7211914, -58.3706665, 74.5044098, -137.4432373, 138.0918579
4: -52.6758118, 88.6924286, -49.1461220, 82.9506073, -135.6264191, 137.8385468

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053808, upper bound: 146.6976578
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7057147, upper bound: 146.6977803
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -69.9139099, 92.8095551, -60.1045952, 78.9650726, -148.8789368, 152.9141541
1: -56.2212448, 76.4052200, -47.6239815, 64.9109879, -121.1322250, 124.0292053
2: -46.1604691, 76.7849960, -39.1903915, 65.7024765, -111.8629379, 115.9753876
3: -73.1382370, 92.7751694, -62.0858307, 78.5961533, -151.7343750, 154.8609924
4: -61.1051254, 102.5622787, -51.9025421, 87.4467010, -148.5518188, 154.4648132

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866627
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6931377, upper bound: 146.6867582
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -60.9523659, 80.0911331, -62.9710464, 82.8379211, -143.7902679, 143.0621796
1: -48.2995453, 65.8306274, -49.9102173, 68.0906219, -116.3901596, 115.7408447
2: -39.7593002, 66.6049957, -41.0812836, 68.9267502, -108.6860504, 107.6862793
3: -62.9388199, 79.7211914, -65.0276794, 82.4494247, -145.3882446, 144.7488708
4: -52.6758118, 88.6924286, -54.4356766, 91.7973709, -144.4731750, 143.1281128

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983757, upper bound: 146.6971143
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6931377, upper bound: 146.6973192
time: 1.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.36 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7060133, upper bound: 146.7059916
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7062746, upper bound: 146.7060347
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7061607, upper bound: 146.7063234
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7064280, upper bound: 146.7064280
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7037085
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7053808
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6873503, upper bound: 146.7037421
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6977803, upper bound: 146.7057147
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7037085, upper bound: 146.6872421
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7035058, upper bound: 146.6873503
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7053808, upper bound: 146.6976578
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.7057147, upper bound: 146.6977803
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866627
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6931377, upper bound: 146.6867582
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6983757, upper bound: 146.6971143
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 0, lower bound: -146.6931377, upper bound: 146.6973192

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -42.4987717, 54.3720055, -58.9967270, 77.3771820, -119.8759537, 113.3687286
1: -32.4521790, 44.8449593, -46.7621422, 64.1594238, -96.6116028, 91.6070709
2: -26.9153976, 45.2453003, -38.6887550, 64.1345520, -91.0499420, 83.9340515
3: -42.4849701, 54.2012787, -60.7557564, 77.6556168, -120.1405869, 114.9570312
4: -35.6614990, 59.9041824, -51.2240829, 85.5531998, -121.2146835, 111.1282425

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7056361, upper bound: 146.7058289
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7056361, upper bound: 146.7055165
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -42.7197113, 54.5723495, -49.9876099, 65.2337418, -107.9534531, 104.5599442
1: -32.6181946, 45.0200043, -39.1246567, 53.8847008, -86.5028839, 84.1446533
2: -27.0563622, 45.3985710, -32.3640900, 54.2850037, -81.3413620, 77.7626572
3: -42.6874313, 54.4159203, -51.0593948, 65.1042557, -107.7916794, 105.4752960
4: -35.8596611, 60.1211357, -42.9750748, 72.2319183, -108.0915833, 103.0961838

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7062746, upper bound: 146.7060347
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7062746, upper bound: 146.7060347
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -47.3947716, 61.1550598, -63.6753807, 83.7255096, -131.1202698, 124.8304443
1: -36.6659050, 50.5002785, -50.6374435, 69.5023499, -106.1682587, 101.1377106
2: -30.3358326, 50.9575005, -41.8704300, 69.4055862, -99.7414169, 92.8279266
3: -47.7369385, 61.0850906, -65.8231430, 84.1494904, -131.8864136, 126.9082336
4: -40.2357483, 67.5837555, -55.4379082, 92.6668625, -132.9026184, 123.0216599

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7060619, upper bound: 146.7060619
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7060619, upper bound: 146.7063234
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -49.6000557, 64.2358017, -54.0333061, 70.6910324, -120.2910843, 118.2691040
1: -38.4503098, 53.0287628, -42.3821487, 58.4163933, -96.8666840, 95.4109039
2: -31.8126259, 53.5952072, -35.0727425, 58.8703728, -90.6829987, 88.6679535
3: -50.0806046, 64.1302261, -55.2978020, 70.5257492, -120.6063538, 119.4280243
4: -42.2183571, 71.1163330, -46.5576744, 78.4373474, -120.6556854, 117.6740112

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7063234, upper bound: 146.7061607
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7063234, upper bound: 146.7064280
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -42.4987717, 54.3720055, -63.9840279, 84.5727615, -127.0715332, 118.3560333
1: -32.4521790, 44.8449593, -51.2528305, 69.5954742, -102.0476532, 96.0977707
2: -26.9153976, 45.2453003, -42.1081886, 69.9556122, -96.8710098, 87.3534851
3: -42.4849701, 54.2012787, -66.5818176, 84.4942093, -126.9791794, 120.7830963
4: -35.6614990, 59.9041824, -55.7171783, 93.3216705, -128.9831696, 115.6213455

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6857108, upper bound: 146.7032732
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6857521, upper bound: 146.7029789
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -42.7197113, 54.5723495, -55.6136246, 72.9487839, -115.6684875, 110.1859741
1: -32.6181946, 45.0200043, -43.9985657, 59.9734993, -92.5916901, 89.0185547
2: -27.0563622, 45.3985710, -36.2178459, 60.6603012, -87.7166595, 81.6164169
3: -42.6874313, 54.4159203, -57.3939819, 72.6465225, -115.3339310, 111.8099060
4: -35.8596611, 60.1211357, -48.0107269, 80.6581345, -116.5177917, 108.1318665

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6956149, upper bound: 146.7053738
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6956149, upper bound: 146.7053808
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.3947716, 61.1550598, -69.2378311, 91.8451691, -139.2399445, 130.3928833
1: -36.6659050, 50.5002785, -55.6464920, 75.6030045, -112.2689056, 106.1467438
2: -30.3358326, 50.9575005, -45.6943398, 75.9636917, -106.2995224, 96.6518402
3: -47.7369385, 61.0850906, -72.3728256, 91.8047028, -139.5416107, 133.4579163
4: -40.2357483, 67.5837555, -60.4851646, 101.4542770, -141.6900024, 128.0689240

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7037085
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7037421
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.6000557, 64.2358017, -60.3397865, 79.2335052, -128.8335571, 124.5755920
1: -38.4503098, 53.0287628, -47.7846985, 65.1170044, -103.5673141, 100.8134613
2: -31.8126259, 53.5952072, -39.3366508, 65.8988190, -97.7114410, 92.9318390
3: -50.0806046, 64.1302261, -62.2575302, 78.8578873, -128.9384918, 126.3877563
4: -42.2183571, 71.1163330, -52.1130943, 87.7264481, -129.9448090, 123.2294159

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6925655, upper bound: 146.7043077
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6925655, upper bound: 146.7057147
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -64.3437653, 85.0824051, -42.4987717, 54.3720055, -118.7157745, 127.5811768
1: -51.5449028, 70.0057907, -32.4521790, 44.8449593, -96.3898468, 102.4579697
2: -42.3501701, 70.3755951, -26.9153976, 45.2453003, -87.5954742, 97.2909927
3: -66.9851456, 84.9909363, -42.4849701, 54.2012787, -121.1864090, 127.4758987
4: -56.0430870, 93.9015808, -35.6614990, 59.9041824, -115.9472656, 129.5630798

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -69.2378311, 91.8451691, -47.3947716, 61.1550598, -130.3928833, 139.2399292
1: -55.6464920, 75.6030045, -36.6659050, 50.5002785, -106.1467438, 112.2689056
2: -45.6943398, 75.9636917, -30.3358326, 50.9575005, -96.6518402, 106.2995224
3: -72.3728256, 91.8047028, -47.7369385, 61.0850906, -133.4579163, 139.5416107
4: -60.4851646, 101.4542770, -40.2357483, 67.5837555, -128.0689240, 141.6900024

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6771886, upper bound: 146.6334222
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7023435, upper bound: 146.6834337
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -55.6136246, 72.9487839, -42.7197113, 54.5723495, -110.1859741, 115.6684875
1: -43.9985657, 59.9734993, -32.6181946, 45.0200043, -89.0185547, 92.5916901
2: -36.2178459, 60.6603012, -27.0563622, 45.3985710, -81.6164169, 87.7166595
3: -57.3939819, 72.6465225, -42.6874313, 54.4159203, -111.8098984, 115.3339310
4: -48.0107269, 80.6581345, -35.8596611, 60.1211357, -108.1318665, 116.5177917

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050194, upper bound: 146.6975519
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050194, upper bound: 146.6976578
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -60.3397865, 79.2335052, -49.6000557, 64.2358017, -124.5755920, 128.8335571
1: -47.7846985, 65.1170044, -38.4503098, 53.0287628, -100.8134613, 103.5673141
2: -39.3366508, 65.8988190, -31.8126259, 53.5952072, -92.9318390, 97.7114410
3: -62.2575302, 78.8578873, -50.0806046, 64.1302261, -126.3877563, 128.9384918
4: -52.1130943, 87.7264481, -42.2183571, 71.1163330, -123.2294159, 129.9448090

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054831, upper bound: 146.6976982
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054831, upper bound: 146.6977803
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -64.3437653, 85.0824051, -45.1360512, 58.6868477, -123.0306015, 130.2184601
1: -51.5449028, 70.0057907, -35.2528992, 48.1175499, -99.6624527, 105.2586899
2: -42.3501701, 70.3755951, -29.0116615, 48.7698174, -91.1199875, 99.3872528
3: -66.9851456, 84.9909363, -46.1453018, 58.3486557, -125.3337936, 131.1361847
4: -56.0430870, 93.9015808, -38.4028168, 64.5153580, -120.5584412, 132.3043823

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6931030, upper bound: 146.6866625
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6931030, upper bound: 146.6866625
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -69.2378311, 91.8451691, -52.9681702, 69.0555191, -138.2933502, 144.8133087
1: -55.6464920, 75.6030045, -41.6567955, 56.7100258, -112.3565063, 117.2597885
2: -45.6943398, 75.9636917, -34.2890282, 57.4796715, -103.1740112, 110.2527161
3: -72.3728256, 91.8047028, -54.1657562, 68.7681732, -141.1409912, 145.9704437
4: -60.4851646, 101.4542770, -45.3937531, 76.2301865, -136.7153473, 146.8479767

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6772032, upper bound: 146.6334217
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6894563, upper bound: 146.6828849
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -55.6136246, 72.9487839, -45.4532700, 59.0495338, -114.6631622, 118.4020538
1: -43.9985657, 59.9734993, -35.5186501, 48.4235001, -92.4220505, 95.4921494
2: -36.2178459, 60.6603012, -29.2362289, 49.0637703, -85.2816162, 89.8965302
3: -57.3939819, 72.6465225, -46.4813843, 58.7150688, -116.1090240, 119.1279068
4: -48.0107269, 80.6581345, -38.7150574, 64.9296417, -112.9403687, 119.3731918

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6970288, upper bound: 146.6970288
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6931030, upper bound: 146.6971143
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -60.3397865, 79.2335052, -55.4923820, 72.5581512, -132.8979187, 134.7258911
1: -47.7846985, 65.1170044, -43.7041435, 59.5614357, -107.3461304, 108.8211517
2: -39.3366508, 65.8988190, -35.9599380, 60.4594040, -99.7960434, 101.8587494
3: -62.2575302, 78.8578873, -56.8300629, 72.2024155, -134.4599152, 135.6879425
4: -52.1130943, 87.7264481, -47.6420860, 80.2286530, -132.3417511, 135.3685303

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6971143, upper bound: 146.6971803
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6971143, upper bound: 146.6973192
time: 1.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.73 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7056361, upper bound: 146.7058289
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7056361, upper bound: 146.7055165
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7062746, upper bound: 146.7060347
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7062746, upper bound: 146.7060347
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7060619, upper bound: 146.7060619
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7060619, upper bound: 146.7063234
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7063234, upper bound: 146.7061607
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7063234, upper bound: 146.7064280
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6857108, upper bound: 146.7032732
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6857521, upper bound: 146.7029789
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6956149, upper bound: 146.7053738
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6956149, upper bound: 146.7053808
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7037085
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7037421
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6925655, upper bound: 146.7043077
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6925655, upper bound: 146.7057147
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6771886, upper bound: 146.6334222
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7023435, upper bound: 146.6834337
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7050194, upper bound: 146.6975519
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7050194, upper bound: 146.6976578
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7054831, upper bound: 146.6976982
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.7054831, upper bound: 146.6977803
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6931030, upper bound: 146.6866625
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6931030, upper bound: 146.6866625
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6772032, upper bound: 146.6334217
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6894563, upper bound: 146.6828849
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6970288, upper bound: 146.6970288
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6931030, upper bound: 146.6971143
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6971143, upper bound: 146.6971803
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.73
Output dim: 0, lower bound: -146.6971143, upper bound: 146.6973192

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -39.5027428, 50.1123734, -57.4053879, 75.1865387, -114.6892700, 107.5177612
1: -29.8666859, 41.3313332, -45.4396820, 62.3205605, -92.1872330, 86.7709732
2: -24.7750511, 41.6043320, -37.6019211, 62.2695961, -87.0446167, 79.2062531
3: -39.1181831, 50.0211487, -59.0208092, 75.4549789, -114.5731583, 109.0419540
4: -32.8657265, 54.9388809, -49.7919502, 83.0248718, -115.8905869, 104.7308273

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053133, upper bound: 146.7057964
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053133, upper bound: 146.7058289
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -65.7173691, 89.3837051, -56.9259453, 74.4110031, -140.1283569, 146.3096161
1: -53.3031883, 75.1622162, -45.0221062, 61.6652527, -114.9684448, 120.1843033
2: -43.7768860, 74.7847366, -37.2667046, 61.5953255, -105.3722000, 112.0514221
3: -70.3007584, 90.6588745, -58.4412270, 74.6532822, -144.9540405, 149.1000977
4: -58.0121460, 100.2484360, -49.3364296, 82.1262283, -140.1383667, 149.5848541

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7054673
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7055165
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -49.9627228, 65.0157700, -49.9876099, 65.2337418, -115.1964645, 115.0033798
1: -39.3158722, 53.5048180, -39.1246567, 53.8847008, -93.2005615, 92.6294708
2: -32.5230789, 53.5936279, -32.3640900, 54.2850037, -86.8080750, 85.9577179
3: -50.9778252, 64.7967606, -51.0593948, 65.1042557, -116.0820694, 115.8561325
4: -43.0311394, 71.2576828, -42.9750748, 72.2319183, -115.2630615, 114.2327499

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7059437
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7060347
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.1203499, 50.7516823, -49.9876099, 65.2337418, -105.3540878, 100.7392883
1: -30.4053383, 41.8726501, -39.1246567, 53.8847008, -84.2900238, 80.9972916
2: -25.2556973, 42.0983925, -32.3640900, 54.2850037, -79.5406799, 74.4624786
3: -39.7605858, 50.6513634, -51.0593948, 65.1042557, -104.8648376, 101.7107391
4: -33.4442749, 55.6784706, -42.9750748, 72.2319183, -105.6761932, 98.6535339

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7059437
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7059916
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -55.2488136, 72.7404633, -63.6753807, 83.7255096, -138.9743042, 136.4158325
1: -43.9266968, 60.0480003, -50.6374435, 69.5023499, -113.4290466, 110.6854401
2: -36.2499580, 60.0269547, -41.8704300, 69.4055862, -105.6555328, 101.8973846
3: -57.1505547, 72.7384491, -65.8231430, 84.1494904, -141.3000488, 138.5615845
4: -48.0576248, 80.1144409, -55.4379082, 92.6668625, -140.7244873, 135.5523376

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7052582
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7057305
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -47.5977097, 61.4924278, -63.6753807, 83.7255096, -131.3231964, 125.1678085
1: -36.8424149, 50.7539215, -50.6374435, 69.5023499, -106.3447571, 101.3913651
2: -30.4794197, 51.2596741, -41.8704300, 69.4055862, -99.8850098, 93.1301041
3: -47.9825516, 61.3918457, -65.8231430, 84.1494904, -132.1320496, 127.2149887
4: -40.4371262, 67.9806137, -55.4379082, 92.6668625, -133.1039886, 123.4184952

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7054633
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7059356
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -56.3390350, 73.6385880, -54.0333061, 70.6910324, -127.0300674, 127.6718903
1: -44.6075058, 61.0470200, -42.3821487, 58.4163933, -103.0238647, 103.4291687
2: -36.9088554, 60.8001595, -35.0727425, 58.8703728, -95.7792282, 95.8729019
3: -57.8986092, 73.9244232, -55.2978020, 70.5257492, -128.4243622, 129.2222137
4: -48.8650551, 81.1451187, -46.5576744, 78.4373474, -127.3023987, 127.7027893

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7060133
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7060721
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.6786270, 61.6097183, -54.0333061, 70.6910324, -118.3696594, 115.6430206
1: -36.9107399, 50.8506050, -42.3821487, 58.4163933, -95.3270874, 93.2327271
2: -30.5371628, 51.3591270, -35.0727425, 58.8703728, -89.4075317, 86.4318695
3: -48.0740280, 61.5079536, -55.2978020, 70.5257492, -118.5997696, 116.8057480
4: -40.5163841, 68.1169739, -46.5576744, 78.4373474, -118.9537354, 114.6746521

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7062746
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7064226
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -39.5027428, 50.1123734, -62.3021736, 82.2566376, -121.7593689, 112.4145508
1: -29.8666859, 41.3313332, -49.8478012, 67.6624451, -97.5291290, 91.1791153
2: -24.7750511, 41.6043320, -40.9638443, 67.9998169, -92.7748489, 82.5681534
3: -39.1181831, 50.0211487, -64.7266006, 82.1700516, -121.2882309, 114.7477417
4: -32.8657265, 54.9388809, -54.2117043, 90.6648178, -123.5305405, 109.1505737

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7024610
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7032734
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -65.7173691, 89.3837051, -62.2734909, 82.0704880, -147.7878265, 151.6571503
1: -53.3031883, 75.1622162, -49.8010216, 67.5321350, -120.8353195, 124.9632111
2: -43.7768860, 74.7847366, -40.9341125, 67.8142395, -111.5911255, 115.7188492
3: -70.3007584, 90.6588745, -64.6471558, 82.0104294, -152.3111877, 155.3060303
4: -58.0121460, 100.2484360, -54.1592674, 90.4388580, -148.4510040, 154.4076843

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7021798
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7029789
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -49.9627228, 65.0157700, -55.6136246, 72.9487839, -122.9115067, 120.6293945
1: -39.3158722, 53.5048180, -43.9985657, 59.9734993, -99.2893677, 97.5033722
2: -32.5230789, 53.5936279, -36.2178459, 60.6603012, -93.1833801, 89.8114777
3: -50.9778252, 64.7967606, -57.3939819, 72.6465225, -123.6243210, 122.1907272
4: -43.0311394, 71.2576828, -48.0107269, 80.6581345, -123.6892700, 119.2684097

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7053634
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7053738
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.1203499, 50.7516823, -55.6136246, 72.9487839, -113.0691299, 106.3653107
1: -30.4053383, 41.8726501, -43.9985657, 59.9734993, -90.3788300, 85.8711929
2: -25.2556973, 42.0983925, -36.2178459, 60.6603012, -85.9159851, 78.3162384
3: -39.7605858, 50.6513634, -57.3939819, 72.6465225, -112.4070892, 108.0453415
4: -33.4442749, 55.6784706, -48.0107269, 80.6581345, -114.1024094, 103.6891937

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7051228
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7051247
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -55.2488136, 72.7404633, -69.2378311, 91.8451691, -147.0939789, 141.9782867
1: -43.9266968, 60.0480003, -55.6464920, 75.6030045, -119.5296860, 115.6944885
2: -36.2499580, 60.0269547, -45.6943398, 75.9636917, -112.2136383, 105.7212982
3: -57.1505547, 72.7384491, -72.3728256, 91.8047028, -148.9552612, 145.1112671
4: -48.0576248, 80.1144409, -60.4851646, 101.4542770, -149.5118713, 140.5996094

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6333936, upper bound: 146.6769057
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6833830, upper bound: 146.7022096
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -47.5977097, 61.4924278, -69.2378311, 91.8451691, -139.4428558, 130.7302551
1: -36.8424149, 50.7539215, -55.6464920, 75.6030045, -112.4453964, 106.4003983
2: -30.4794197, 51.2596741, -45.6943398, 75.9636917, -106.4431152, 96.9540100
3: -47.9825516, 61.3918457, -72.3728256, 91.8047028, -139.7872620, 133.7646790
4: -40.4371262, 67.9806137, -60.4851646, 101.4542770, -141.8913574, 128.4657745

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6333936, upper bound: 146.6771886
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6833830, upper bound: 146.7023435
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -56.3390350, 73.6385880, -60.3397865, 79.2335052, -135.5725403, 133.9783783
1: -44.6075058, 61.0470200, -47.7846985, 65.1170044, -109.7245102, 108.8317184
2: -36.9088554, 60.8001595, -39.3366508, 65.8988190, -102.8076782, 100.1368027
3: -57.8986092, 73.9244232, -62.2575302, 78.8578873, -136.7565002, 136.1819458
4: -48.8650551, 81.1451187, -52.1130943, 87.7264481, -136.5915070, 133.2582092

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7043077
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7043077
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.6786270, 61.6097183, -60.3397865, 79.2335052, -126.9121323, 121.9495087
1: -36.9107399, 50.8506050, -47.7846985, 65.1170044, -102.0277328, 98.6352921
2: -30.5371628, 51.3591270, -39.3366508, 65.8988190, -96.4359818, 90.6957703
3: -48.0740280, 61.5079536, -62.2575302, 78.8578873, -126.9319153, 123.7654877
4: -40.5163841, 68.1169739, -52.1130943, 87.7264481, -128.2428284, 120.2300568

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7054831
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7056155
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -71.2851715, 94.6299820, -41.3181648, 52.6793365, -123.9645081, 135.9481201
1: -56.5412025, 77.9430542, -31.3937416, 43.5182953, -100.0594940, 109.3367920
2: -46.7002220, 78.8790665, -26.0572777, 43.9234734, -90.6236954, 104.9363327
3: -73.7618866, 94.1654663, -41.1692619, 52.6074142, -126.3692856, 135.3347168
4: -61.9301300, 105.2862091, -34.5454178, 58.1006775, -120.0307999, 139.8316345

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -60.0611649, 79.1677170, -41.7063484, 53.1932716, -113.2544327, 120.8740540
1: -47.8753433, 64.9970703, -31.6974506, 43.8273964, -91.7027283, 96.6945038
2: -39.3818207, 65.4934616, -26.3029537, 44.2486877, -83.6304932, 91.7964096
3: -62.2327843, 78.8950500, -41.4820862, 52.9675636, -115.2003479, 120.3771362
4: -52.1069756, 87.3030853, -34.8364334, 58.5443192, -110.6512909, 122.1395187

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -78.6598434, 104.7793884, -46.1815796, 59.3626556, -138.0224915, 150.9609680
1: -62.6864204, 86.3446045, -35.5666046, 49.0048180, -111.6912384, 121.9112091
2: -51.7383270, 87.3350754, -29.4516945, 49.4479523, -101.1862793, 116.7867737
3: -81.7546539, 104.3666382, -46.3091736, 59.2808342, -141.0354919, 150.6757660
4: -68.5973663, 116.7258606, -39.0616684, 65.5230408, -134.1203918, 155.7875061

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6769057, upper bound: 146.6333936
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6769057, upper bound: 146.6334222
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -64.9658508, 85.9501801, -46.4293900, 59.7030792, -124.6689301, 132.3795624
1: -51.9960442, 70.6328888, -35.7401886, 49.2671585, -101.2631989, 106.3730774
2: -42.7397270, 71.0999146, -29.5946217, 49.7572937, -92.4970093, 100.6945343
3: -67.6466370, 85.7477036, -46.5405693, 59.5911942, -127.2378311, 132.2882690
4: -56.5694389, 94.8905563, -39.2490273, 65.9520111, -122.5214462, 134.1395874

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020471, upper bound: 146.6833830
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7022096, upper bound: 146.6834337
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -42.7197113, 54.5723495, -97.1784668, 97.6207581
1: -33.1106224, 45.0279160, -32.6181946, 45.0200043, -78.1306305, 77.6461029
2: -27.2823734, 45.4918365, -27.0563622, 45.3985710, -72.6809464, 72.5481949
3: -43.3220978, 54.6532707, -42.6874313, 54.4159203, -97.7380219, 97.3406982
4: -36.1003265, 60.1447449, -35.8596611, 60.1211357, -96.2214508, 96.0044098

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6956149
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6975519
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.7565994, 70.2050858, -42.7197113, 54.5723495, -108.3289261, 112.9247894
1: -42.3193741, 57.6221886, -32.6181946, 45.0200043, -87.3393631, 90.2403870
2: -34.8279877, 58.4526215, -27.0563622, 45.3985710, -80.2265625, 85.5089874
3: -55.0447731, 69.8646393, -42.6874313, 54.4159203, -109.4606781, 112.5520706
4: -46.1298752, 77.5479279, -35.8596611, 60.1211357, -106.2510071, 113.4075928

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6956149
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6976578
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -49.6000557, 64.2358017, -106.8419189, 104.5010986
1: -33.1106224, 45.0279160, -38.4503098, 53.0287628, -86.1393890, 83.4782181
2: -27.2823734, 45.4918365, -31.8126259, 53.5952072, -80.8775787, 77.3044586
3: -43.3220978, 54.6532707, -50.0806046, 64.1302261, -107.4523239, 104.7338715
4: -36.1003265, 60.1447449, -42.2183571, 71.1163330, -107.2166595, 102.3630981

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6925367
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6976807
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -53.7659836, 70.2163086, -49.6000557, 64.2358017, -118.0017853, 119.8163605
1: -42.3269310, 57.6318207, -38.4503098, 53.0287628, -95.3556900, 96.0821304
2: -34.8345337, 58.4619865, -31.8126259, 53.5952072, -88.4297409, 90.2746124
3: -55.0541573, 69.8761063, -50.0806046, 64.1302261, -119.1843872, 119.9567108
4: -46.1383362, 77.5608292, -42.2183571, 71.1163330, -117.2546692, 119.7791672

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6925655
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6977145
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -54.8935204, 71.9935226, -45.1360512, 58.6868477, -113.5803604, 117.1295700
1: -43.7559738, 59.0566177, -35.2528992, 48.1175499, -91.8735199, 94.3095169
2: -35.9394417, 59.3323936, -29.0116615, 48.7698174, -84.7092590, 88.3440475
3: -56.7730713, 71.7635269, -46.1453018, 58.3486557, -115.1217270, 117.9088287
4: -47.5056877, 78.9019928, -38.4028168, 64.5153580, -112.0210419, 117.3048096

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6865776
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866625
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -61.7200356, 81.1029663, -45.1360512, 58.6868477, -120.4068832, 126.2390137
1: -49.2398033, 66.7187576, -35.2528992, 48.1175499, -97.3573532, 101.9716568
2: -40.5049629, 66.9290161, -29.0116615, 48.7698174, -89.2747803, 95.9406738
3: -63.8531952, 81.0615082, -46.1453018, 58.3486557, -122.2018509, 127.2068100
4: -53.5745544, 89.2021790, -38.4028168, 64.5153580, -118.0898895, 127.6049957

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6865776
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866625
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -78.6598434, 104.7793884, -51.3967667, 66.8368988, -145.4967194, 156.1761475
1: -62.6864204, 86.3446045, -40.2936172, 54.8909912, -117.5774078, 126.6382141
2: -51.7383270, 87.3350754, -33.1915398, 55.6304131, -107.3687439, 120.5266113
3: -81.7546539, 104.3666382, -52.4004784, 66.5631866, -148.3178406, 156.7670898
4: -68.5973663, 116.7258606, -43.9413872, 73.7226639, -142.3200378, 160.6672211

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6769102, upper bound: 146.6333926
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6769102, upper bound: 146.6334217
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -64.9658508, 85.9501801, -52.0283241, 67.7388687, -132.7047119, 137.9785004
1: -51.9960442, 70.6328888, -40.8161049, 55.5672874, -107.5633316, 111.4489899
2: -42.7397270, 71.0999146, -33.6053200, 56.3716888, -99.1114120, 104.7052307
3: -67.6466370, 85.7477036, -53.0618744, 67.3721466, -135.0187836, 138.8095703
4: -56.5694389, 94.8905563, -44.4806709, 74.7328491, -131.3022919, 139.3712311

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6827820, upper bound: 146.6827820
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6827820, upper bound: 146.6827820
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -45.4532700, 59.0495338, -101.6556549, 100.3543091
1: -33.1106224, 45.0279160, -35.5186501, 48.4235001, -81.5341187, 80.5465546
2: -27.2823734, 45.4918365, -29.2362289, 49.0637703, -76.3461456, 74.7280655
3: -43.3220978, 54.6532707, -46.4813843, 58.7150688, -102.0371399, 101.1346588
4: -36.1003265, 60.1447449, -38.7150574, 64.9296417, -101.0299606, 98.8598022

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6948933
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6970288
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.7565994, 70.2050858, -45.4532700, 59.0495338, -112.8061371, 115.6583481
1: -42.3193741, 57.6221886, -35.5186501, 48.4235001, -90.7428589, 93.1408386
2: -34.8279877, 58.4526215, -29.2362289, 49.0637703, -83.8917542, 87.6888504
3: -55.0447731, 69.8646393, -46.4813843, 58.7150688, -113.7597961, 116.3460236
4: -46.1298752, 77.5479279, -38.7150574, 64.9296417, -111.0595093, 116.2629852

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6948933
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6971143
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -55.4923820, 72.5581512, -115.1642761, 110.3934250
1: -33.1106224, 45.0279160, -43.7041435, 59.5614357, -92.6720581, 88.7320557
2: -27.2823734, 45.4918365, -35.9599380, 60.4594040, -87.7417755, 81.4517670
3: -43.3220978, 54.6532707, -56.8300629, 72.2024155, -115.5245056, 111.4833374
4: -36.1003265, 60.1447449, -47.6420860, 80.2286530, -116.3289719, 107.7868347

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6925367
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6971360
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -53.7659836, 70.2163086, -55.4923820, 72.5581512, -126.3241272, 125.7086945
1: -42.3269310, 57.6318207, -43.7041435, 59.5614357, -101.8883667, 101.3359680
2: -34.8345337, 58.4619865, -35.9599380, 60.4594040, -95.2939377, 94.4219208
3: -55.0541573, 69.8761063, -56.8300629, 72.2024155, -127.2565765, 126.7061691
4: -46.1383362, 77.5608292, -47.6420860, 80.2286530, -126.3669891, 125.2029114

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6925655
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6971897
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.23 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053133, upper bound: 146.7057964
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053133, upper bound: 146.7058289
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7054673
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7055165
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7059437
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7060347
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7059437
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7057717, upper bound: 146.7059916
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7052582
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7057305
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7054633
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6981945, upper bound: 146.7059356
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7060133
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7060721
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7062746
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7058203, upper bound: 146.7064226
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7024610
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7032734
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7021798
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7029789
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7053634
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7053738
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7051228
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6872421, upper bound: 146.7051247
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6333936, upper bound: 146.6769057
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6833830, upper bound: 146.7022096
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6333936, upper bound: 146.6771886
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6833830, upper bound: 146.7023435
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7043077
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7043077
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7054831
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6873117, upper bound: 146.7056155
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6769057, upper bound: 146.6333936
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6769057, upper bound: 146.6334222
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7020471, upper bound: 146.6833830
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7022096, upper bound: 146.6834337
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6956149
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6975519
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6956149
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7053634, upper bound: 146.6976578
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6925367
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6976807
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6925655
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.7043079, upper bound: 146.6977145
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6865776
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866625
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6865776
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866625
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6769102, upper bound: 146.6333926
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6769102, upper bound: 146.6334217
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6827820, upper bound: 146.6827820
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6827820, upper bound: 146.6827820
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6948933
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6970288
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6948933
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6954881, upper bound: 146.6971143
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6925367
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6971360
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6925655
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -146.6866627, upper bound: 146.6971897

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -39.5027428, 50.1123734, -50.8785782, 66.2578201, -105.7605591, 100.9909515
1: -29.8666859, 41.3313332, -40.1250687, 54.6600418, -84.5267181, 81.4563751
2: -24.7750511, 41.6043320, -33.1862373, 54.6279526, -79.4029922, 74.7905502
3: -39.1181831, 50.0211487, -52.0260620, 66.2380447, -105.3562164, 102.0472107
4: -32.8657265, 54.9388809, -43.9062767, 72.6357880, -105.5015030, 98.8451385

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052988, upper bound: 146.7057964
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052988, upper bound: 146.7057964
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -39.5027428, 50.1123734, -54.0057068, 70.2701187, -109.7728577, 104.1180801
1: -29.8666859, 41.3313332, -42.5968475, 58.2333679, -88.1000519, 83.9281387
2: -24.7750511, 41.6043320, -35.2764168, 57.9389076, -82.7139130, 76.8807297
3: -39.1181831, 50.0211487, -55.2251091, 70.5560760, -109.6742477, 105.2462463
4: -32.8657265, 54.9388809, -46.6973114, 77.2359924, -110.1017151, 101.6361847

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052988, upper bound: 146.7058289
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052988, upper bound: 146.7058289
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -65.7173691, 89.3837051, -50.3345146, 65.4145355, -131.1318970, 139.7182159
1: -53.3031883, 75.1622162, -39.6695480, 53.9503517, -107.2535400, 114.8317490
2: -43.7768860, 74.7847366, -32.8276215, 53.8980026, -97.6748886, 107.6123505
3: -70.3007584, 90.6588745, -51.4088402, 65.3840027, -135.6847534, 142.0677032
4: -58.0121460, 100.2484360, -43.4155045, 71.6692810, -129.6814270, 143.6639404

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7054673
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7054673
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -65.7173691, 89.3837051, -55.3380814, 72.2211609, -137.9384918, 144.7217560
1: -53.3031883, 75.1622162, -43.7399139, 59.8254318, -113.1286163, 118.9021301
2: -43.7768860, 74.7847366, -36.1979446, 59.6116066, -103.3884888, 110.9826813
3: -70.3007584, 90.6588745, -56.7479668, 72.4645996, -142.7653503, 147.4068451
4: -58.0121460, 100.2484360, -47.9152031, 79.5164032, -137.5285492, 148.1636200

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7055165
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7055165
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -49.9627228, 65.0157700, -40.1203499, 50.7516823, -100.7144012, 105.1361237
1: -39.3158722, 53.5048180, -30.4053383, 41.8726501, -81.1885223, 83.9101562
2: -32.5230789, 53.5936279, -25.2556973, 42.0983925, -74.6214752, 78.8493118
3: -50.9778252, 64.7967606, -39.7605858, 50.6513634, -101.6291733, 104.5573349
4: -43.0311394, 71.2576828, -33.4442749, 55.6784706, -98.7096100, 104.7019577

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052988, upper bound: 146.7057982
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7054673
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -49.9627228, 65.0157700, -47.6759415, 61.6059189, -111.5686340, 112.6917114
1: -39.3158722, 53.5048180, -36.9085884, 50.8474579, -90.1633301, 90.4134064
2: -32.5230789, 53.5936279, -30.5354347, 51.3558693, -83.8789444, 84.1290588
3: -50.9778252, 64.7967606, -48.0711441, 61.5041656, -112.4819717, 112.8679047
4: -43.0311394, 71.2576828, -40.5140724, 68.1126556, -111.1437988, 111.7717590

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052988, upper bound: 146.7058578
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7055778
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -40.1203499, 50.7516823, -40.1203499, 50.7516823, -90.8720322, 90.8720322
1: -30.4053383, 41.8726501, -30.4053383, 41.8726501, -72.2779846, 72.2779846
2: -25.2556973, 42.0983925, -25.2556973, 42.0983925, -67.3540878, 67.3540878
3: -39.7605858, 50.6513634, -39.7605858, 50.6513634, -90.4119339, 90.4119415
4: -33.4442749, 55.6784706, -33.4442749, 55.6784706, -89.1227417, 89.1227417

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053133, upper bound: 146.7057964
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7054673
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -40.1203499, 50.7516823, -47.6759415, 61.6059189, -101.7262573, 98.4276276
1: -30.4053383, 41.8726501, -36.9085884, 50.8474579, -81.2527924, 78.7812271
2: -25.2556973, 42.0983925, -30.5354347, 51.3558693, -76.6115570, 72.6338272
3: -39.7605858, 50.6513634, -48.0711441, 61.5041656, -101.2647400, 98.7225037
4: -33.4442749, 55.6784706, -40.5140724, 68.1126556, -101.5569305, 96.1925430

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053133, upper bound: 146.7058289
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053531, upper bound: 146.7055165
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -52.9075661, 69.3590317, -58.4482307, 76.4898529, -129.3974152, 127.8072662
1: -41.9247856, 57.2563705, -46.3109550, 63.4572525, -105.3820343, 103.5673218
2: -34.6298332, 57.1513405, -38.3117523, 63.2759514, -97.9057846, 95.4630814
3: -54.4970932, 69.3825607, -60.1245117, 76.9117126, -131.4088135, 129.5070648
4: -45.9096031, 76.2104034, -50.7385559, 84.3739243, -130.2835236, 126.9489594

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6977222, upper bound: 146.6977222
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6977222, upper bound: 146.7052582
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -54.2627029, 71.3824615, -80.7480698, 108.5430374, -162.8057098, 152.1304932
1: -43.1083298, 58.8960114, -65.4479599, 91.4368286, -134.5451660, 124.3439713
2: -35.5781059, 58.8748550, -53.8956947, 90.6747437, -126.2528534, 112.7705307
3: -56.0774727, 71.3528137, -85.6893616, 110.4650497, -166.5425262, 157.0421295
4: -47.1662788, 78.5663834, -71.3353271, 121.5892410, -168.7555237, 149.9017029

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052582, upper bound: 146.6981945
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052582, upper bound: 146.7057305
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -45.4908524, 58.3691559, -58.4482307, 76.4898529, -121.9807053, 116.8173828
1: -35.0139542, 48.1934929, -46.3109550, 63.4572525, -98.4712067, 94.5044479
2: -28.9839764, 48.5618668, -38.3117523, 63.2759514, -92.2599182, 86.8736191
3: -45.5518227, 58.3484612, -60.1245117, 76.9117126, -122.4635315, 118.4729767
4: -38.4672050, 64.2991791, -50.7385559, 84.3739243, -122.8411255, 115.0377350

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6980965, upper bound: 146.7045606
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6977222, upper bound: 146.7054633
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.5124702, 59.9244690, -80.7480698, 108.5430374, -155.0554657, 140.6725159
1: -35.8752708, 49.4337959, -65.4479599, 91.4368286, -127.3121033, 114.8817596
2: -29.6893997, 49.9354134, -53.8956947, 90.6747437, -120.3641434, 103.8310776
3: -46.7214012, 59.8088379, -85.6893616, 110.4650497, -157.1864471, 145.4981995
4: -39.3894768, 66.1655884, -71.3353271, 121.5892410, -160.9787140, 137.5009155

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054863, upper bound: 146.7048649
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054863, upper bound: 146.7059356
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -56.3390350, 73.6385880, -40.1203499, 50.7516823, -107.0907135, 113.7589264
1: -44.6075058, 61.0470200, -30.4053383, 41.8726501, -86.4801331, 91.4523621
2: -36.9088554, 60.8001595, -25.2556973, 42.0983925, -79.0072479, 86.0558548
3: -57.8986092, 73.9244232, -39.7605858, 50.6513634, -108.5499725, 113.6850128
4: -48.8650551, 81.1451187, -33.4442749, 55.6784706, -104.5435257, 114.5893936

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049301, upper bound: 146.6981452
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054024, upper bound: 146.7056812
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -56.3390350, 73.6385880, -47.6786270, 61.6097183, -117.9487534, 121.3172073
1: -44.6075058, 61.0470200, -36.9107399, 50.8506050, -95.4580841, 97.9577484
2: -36.9088554, 60.8001595, -30.5371628, 51.3591270, -88.2679825, 91.3373260
3: -57.8986092, 73.9244232, -48.0740280, 61.5079536, -119.4065552, 121.9984436
4: -48.8650551, 81.1451187, -40.5163841, 68.1169739, -116.9820251, 121.6614990

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049301, upper bound: 146.6981452
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049301, upper bound: 146.7057338
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.6786270, 61.6097183, -40.1203499, 50.7516823, -98.4303131, 101.7300720
1: -36.9107399, 50.8506050, -30.4053383, 41.8726501, -78.7833557, 81.2559357
2: -30.5371628, 51.3591270, -25.2556973, 42.0983925, -72.6355591, 76.6148224
3: -48.0740280, 61.5079536, -39.7605858, 50.6513634, -98.7253723, 101.2685242
4: -40.5163841, 68.1169739, -33.4442749, 55.6784706, -96.1948547, 101.5612488

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052694, upper bound: 146.7049802
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054858, upper bound: 146.7058863
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.6786270, 61.6097183, -47.6786270, 61.6097183, -109.2883453, 109.2883453
1: -36.9107399, 50.8506050, -36.9107399, 50.8506050, -87.7613068, 87.7613068
2: -30.5371628, 51.3591270, -30.5371628, 51.3591270, -81.8962860, 81.8962860
3: -48.0740280, 61.5079536, -48.0740280, 61.5079536, -109.5819626, 109.5819626
4: -40.5163841, 68.1169739, -40.5163841, 68.1169739, -108.6333618, 108.6333618

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052694, upper bound: 146.7049802
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054858, upper bound: 146.7059820
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -39.5027428, 50.1123734, -53.4029274, 69.9039993, -109.4067383, 103.5153046
1: -29.8666859, 41.3313332, -42.5069427, 57.3562546, -87.2229385, 83.8382492
2: -24.7750511, 41.6043320, -34.9159889, 57.5735550, -82.3485870, 76.5202942
3: -39.1181831, 50.0211487, -55.1083870, 69.7196503, -108.8378296, 105.1295319
4: -32.8657265, 54.9388809, -46.1588249, 76.4895554, -109.3552704, 101.0977020

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7024610
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7024610
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -39.5027428, 50.1123734, -58.9594955, 77.1854553, -116.6881866, 109.0718536
1: -29.8666859, 41.3313332, -46.9215965, 63.4884872, -93.3551712, 88.2528839
2: -24.7750511, 41.6043320, -38.6228867, 63.6122437, -88.3872757, 80.2271957
3: -39.1181831, 50.0211487, -60.7639542, 77.1702881, -116.2884598, 110.7850876
4: -32.8657265, 54.9388809, -51.0811272, 84.6960068, -117.5617218, 106.0199814

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7032734
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798023, upper bound: 146.7032732
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -65.7173691, 89.3837051, -53.0486412, 69.2588654, -134.9762268, 142.4323120
1: -53.3031883, 75.1622162, -42.1966133, 56.8166428, -110.1198273, 117.3588104
2: -43.7768860, 74.7847366, -34.6760635, 56.9939384, -100.7708282, 109.4607925
3: -70.3007584, 90.6588745, -54.6623993, 69.0862961, -139.3870392, 145.3212433
4: -58.0121460, 100.2484360, -45.8175583, 75.7282715, -133.7404175, 146.0659790

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7021798
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7021798
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -65.7173691, 89.3837051, -60.6851807, 79.6648178, -145.3821869, 150.0688324
1: -53.3031883, 75.1622162, -48.3846817, 65.5302277, -118.8334122, 123.5468979
2: -43.7768860, 74.7847366, -39.8063545, 65.7196198, -109.4965057, 114.5910873
3: -70.3007584, 90.6588745, -62.7150421, 79.6260529, -149.9268188, 153.3739166
4: -58.0121460, 100.2484360, -52.6437950, 87.5665894, -145.5787354, 152.8922119

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798430, upper bound: 146.7029789
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798429, upper bound: 146.7029789
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -49.9627228, 65.0157700, -42.6061211, 54.9010429, -104.8637695, 107.6218872
1: -39.3158722, 53.5048180, -33.1106224, 45.0279160, -84.3437881, 86.6154404
2: -32.5230789, 53.5936279, -27.2823734, 45.4918365, -78.0149078, 80.8759995
3: -50.9778252, 64.7967606, -43.3220978, 54.6532707, -105.6310883, 108.1188507
4: -43.0311394, 71.2576828, -36.1003265, 60.1447449, -103.1758881, 107.3580093

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6879665, upper bound: 146.7018247
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6938412, upper bound: 146.7043081
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -49.9627228, 65.0157700, -53.7565994, 70.2050858, -120.1678085, 118.7723694
1: -39.3158722, 53.5048180, -42.3193741, 57.6221886, -96.9380646, 95.8241806
2: -32.5230789, 53.5936279, -34.8279877, 58.4526215, -90.9757004, 88.4216156
3: -50.9778252, 64.7967606, -55.0447731, 69.8646393, -120.8424530, 119.8414993
4: -43.0311394, 71.2576828, -46.1298752, 77.5479279, -120.5790710, 117.3875580

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6879665, upper bound: 146.7018249
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6938412, upper bound: 146.7043237
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -40.1203499, 50.7516823, -42.6061211, 54.9010429, -95.0213928, 93.3578033
1: -30.4053383, 41.8726501, -33.1106224, 45.0279160, -75.4332504, 74.9832687
2: -25.2556973, 42.0983925, -27.2823734, 45.4918365, -70.7475128, 69.3807678
3: -39.7605858, 50.6513634, -43.3220978, 54.6532707, -94.4138565, 93.9734650
4: -33.4442749, 55.6784706, -36.1003265, 60.1447449, -93.5890198, 91.7787933

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6925724, upper bound: 146.7036204
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6946271, upper bound: 146.7036481
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -40.1203499, 50.7516823, -53.7565994, 70.2050858, -110.3254318, 104.5082779
1: -30.4053383, 41.8726501, -42.3193741, 57.6221886, -88.0275269, 84.1920013
2: -25.2556973, 42.0983925, -34.8279877, 58.4526215, -83.7083206, 76.9263763
3: -39.7605858, 50.6513634, -55.0447731, 69.8646393, -109.6252213, 105.6961136
4: -33.4442749, 55.6784706, -46.1298752, 77.5479279, -110.9922028, 101.8083496

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6925724, upper bound: 146.7036204
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6946271, upper bound: 146.7036715
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -52.0786858, 68.1946335, -78.6598434, 104.7793884, -156.8580475, 146.8544769
1: -41.0925140, 56.2148819, -62.6864204, 86.3446045, -127.4370956, 118.9012909
2: -33.9962730, 56.3007240, -51.7383270, 87.3350754, -121.3313446, 108.0390472
3: -53.4247093, 68.0130615, -81.7546539, 104.3666382, -157.7913055, 149.7677155
4: -45.0631752, 75.0740204, -68.5973663, 116.7258606, -161.7889862, 143.6713867

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -54.6387558, 71.9012146, -64.9658508, 85.9501801, -140.5889282, 136.8670654
1: -43.4060555, 59.3334198, -51.9960442, 70.6328888, -114.0389404, 111.3294601
2: -35.8258171, 59.3257980, -42.7397270, 71.0999146, -106.9257202, 102.0655136
3: -56.4711723, 71.8708725, -67.6466370, 85.7477036, -142.2188721, 139.5175171
4: -47.4974937, 79.1711884, -56.5694389, 94.8905563, -142.3880463, 135.7406311

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6822163, upper bound: 146.6952842
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6826319, upper bound: 146.7017355
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.3015900, 59.5704727, -78.6598434, 104.7793884, -151.0809631, 138.2303162
1: -35.6646042, 49.1569633, -62.6864204, 86.3446045, -122.0092010, 111.8433838
2: -29.5369091, 49.6503258, -51.7383270, 87.3350754, -116.8719864, 101.3886566
3: -46.4402237, 59.4580536, -81.7546539, 104.3666382, -150.8068237, 141.2127075
4: -39.1790352, 65.7809753, -68.5973663, 116.7258606, -155.9048920, 134.3783417

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.6878433, 60.1091156, -64.9658508, 85.9501801, -132.6380157, 125.0749664
1: -35.9544144, 49.5733261, -51.9960442, 70.6328888, -106.5872879, 101.5693665
2: -29.7715626, 50.1150818, -42.7397270, 71.0999146, -100.8714752, 92.8548050
3: -46.8386383, 59.9621506, -67.6466370, 85.7477036, -132.5863342, 127.6087875
4: -39.4964180, 66.4241486, -56.5694389, 94.8905563, -134.3869781, 122.9935913

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6825294, upper bound: 146.7014362
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6825294, upper bound: 146.7019656
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -56.3390350, 73.6385880, -42.6061211, 54.9010429, -111.2400818, 116.2447052
1: -44.6075058, 61.0470200, -33.1106224, 45.0279160, -89.6353989, 94.1576385
2: -36.9088554, 60.8001595, -27.2823734, 45.4918365, -82.4006882, 88.0825348
3: -57.8986092, 73.9244232, -43.3220978, 54.6532707, -112.5518799, 117.2465210
4: -48.8650551, 81.1451187, -36.1003265, 60.1447449, -109.0097961, 117.2454376

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6724594, upper bound: 146.6972431
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6925367, upper bound: 146.7043079
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -56.3390350, 73.6385880, -53.7659836, 70.2163086, -126.5553436, 127.4045486
1: -44.6075058, 61.0470200, -42.3269310, 57.6318207, -102.2393112, 103.3739471
2: -36.9088554, 60.8001595, -34.8345337, 58.4619865, -95.3708344, 95.6346893
3: -57.8986092, 73.9244232, -55.0541573, 69.8761063, -127.7747192, 128.9785767
4: -48.8650551, 81.1451187, -46.1383362, 77.5608292, -126.4258881, 127.2834549

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6724594, upper bound: 146.6972431
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6925367, upper bound: 146.7043077
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.6786270, 61.6097183, -42.6061211, 54.9010429, -102.5796661, 104.2158356
1: -36.9107399, 50.8506050, -33.1106224, 45.0279160, -81.9386215, 83.9612198
2: -30.5371628, 51.3591270, -27.2823734, 45.4918365, -76.0289917, 78.6415024
3: -48.0740280, 61.5079536, -43.3220978, 54.6532707, -102.7272949, 104.8300476
4: -40.5163841, 68.1169739, -36.1003265, 60.1447449, -100.6611328, 104.2173004

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6904763, upper bound: 146.7029717
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976982, upper bound: 146.7054831
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.6786270, 61.6097183, -53.7659836, 70.2163086, -117.8949356, 115.3757019
1: -36.9107399, 50.8506050, -42.3269310, 57.6318207, -94.5425415, 93.1775131
2: -30.5371628, 51.3591270, -34.8345337, 58.4619865, -88.9991302, 86.1936646
3: -48.0740280, 61.5079536, -55.0541573, 69.8761063, -117.9501266, 116.5621109
4: -40.5163841, 68.1169739, -46.1383362, 77.5608292, -118.0772095, 114.2553101

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6904763, upper bound: 146.7029717
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976982, upper bound: 146.7056045
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -71.2851715, 94.6299820, -47.0067444, 60.8054695, -132.0906067, 141.6367035
1: -56.5412025, 77.9430542, -36.6292343, 49.9392891, -106.4804916, 114.5722885
2: -46.7002220, 78.8790665, -30.3801460, 50.1551208, -96.8553467, 109.2592163
3: -73.7618866, 94.1654663, -47.5090446, 60.4205208, -134.1824036, 141.6744690
4: -61.9301300, 105.2862091, -40.1909370, 66.6122665, -128.5423737, 145.4771423

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -71.2851715, 94.6299820, -38.8704376, 48.9268265, -120.2119904, 133.5004120
1: -56.5412025, 77.9430542, -29.2801056, 40.4384804, -96.9796829, 107.2231598
2: -46.7002220, 78.8790665, -24.3400459, 40.6705666, -87.3707886, 103.2191162
3: -73.7618866, 94.1654663, -38.3529282, 48.9322739, -122.6941605, 132.5183868
4: -61.9301300, 105.2862091, -32.2520218, 53.7338219, -115.6639404, 137.5382080

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6332313
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -60.0611649, 79.1677170, -49.5731392, 64.4689713, -124.5301361, 128.7408447
1: -47.8753433, 64.9970703, -38.9768295, 53.0419006, -100.9172211, 103.9738693
2: -39.3818207, 65.4934616, -32.2463684, 53.1365585, -92.5183563, 97.7398300
3: -62.2327843, 78.8950500, -50.5364227, 64.2347641, -126.4675446, 129.4314728
4: -52.1069756, 87.3030853, -42.6650810, 70.6425323, -122.7494965, 129.9681702

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754047, upper bound: 146.6833210
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -60.0611649, 79.1677170, -39.4238205, 49.6962433, -109.7574081, 118.5915222
1: -47.8753433, 64.9970703, -29.7293835, 40.9530182, -88.8283386, 94.7264481
2: -39.3818207, 65.4934616, -24.7063980, 41.2030220, -80.5848083, 90.1998596
3: -62.2327843, 78.8950500, -38.8569756, 49.5387802, -111.7715607, 117.7520218
4: -52.1069756, 87.3030853, -32.7036819, 54.4591522, -106.5661316, 120.0067596

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020562, upper bound: 146.6833210
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -78.6598434, 104.7793884, -51.8414574, 67.7782822, -146.4381256, 156.6208344
1: -62.6864204, 86.3446045, -40.8265076, 55.8671761, -118.5535965, 127.1711121
2: -51.7383270, 87.3350754, -33.7992935, 55.9578667, -107.6961975, 121.1343689
3: -81.7546539, 104.3666382, -53.0739861, 67.5763779, -149.3310242, 157.4405975
4: -68.5973663, 116.7258606, -44.7999840, 74.6070709, -143.2044373, 161.5258331

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6768451, upper bound: 146.6333724
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754045, upper bound: 146.6333936
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754045, upper bound: 146.6332313
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -78.6598434, 104.7793884, -46.2972603, 59.5622520, -138.2220764, 151.0766296
1: -62.6864204, 86.3446045, -35.6593170, 49.1501350, -111.8365555, 122.0039139
2: -51.7383270, 87.3350754, -29.5330677, 49.6435127, -101.3818359, 116.8681412
3: -81.7546539, 104.3666382, -46.4331818, 59.4495773, -141.2042236, 150.7997894
4: -68.5973663, 116.7258606, -39.1738396, 65.7717285, -134.3690948, 155.8996277

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6768451, upper bound: 146.6334010
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754045, upper bound: 146.6334222
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754045, upper bound: 146.6332313
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -64.9658508, 85.9501801, -53.6127739, 70.3195572, -135.2854004, 139.5629120
1: -51.9960442, 70.6328888, -42.4599419, 58.0332031, -110.0292282, 113.0928345
2: -42.7397270, 71.0999146, -35.0846672, 58.0005226, -100.7402420, 106.1845856
3: -67.6466370, 85.7477036, -55.1753273, 70.2763062, -137.9229431, 140.9230347
4: -56.5694389, 94.8905563, -46.5007439, 77.3695374, -133.9389801, 141.3912964

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020471, upper bound: 146.6833830
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020471, upper bound: 146.6833210
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -64.9658508, 85.9501801, -46.6878395, 60.1091003, -125.0749512, 132.6380157
1: -51.9960442, 70.6328888, -35.9544144, 49.5733185, -101.5693588, 106.5872879
2: -42.7397270, 71.0999146, -29.7715588, 50.1150780, -92.8547974, 100.8714752
3: -67.6466370, 85.7477036, -46.8386269, 59.9621468, -127.6087799, 132.5863342
4: -56.5694389, 94.8905563, -39.4964142, 66.4241409, -122.9935760, 134.3869629

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020471, upper bound: 146.6834337
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7020471, upper bound: 146.6833210
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -49.9627228, 65.0157700, -107.6218872, 104.8637695
1: -33.1106224, 45.0279160, -39.3158722, 53.5048180, -86.6154404, 84.3437881
2: -27.2823734, 45.4918365, -32.5230789, 53.5936279, -80.8759995, 78.0149078
3: -43.3220978, 54.6532707, -50.9778252, 64.7967606, -108.1188583, 105.6310883
4: -36.1003265, 60.1447449, -43.0311394, 71.2576828, -107.3580093, 103.1758881

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7044574, upper bound: 146.6936950
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7044302, upper bound: 146.6974981
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7044302, upper bound: 146.6980158
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -40.1203499, 50.7516823, -93.3578033, 95.0213928
1: -33.1106224, 45.0279160, -30.4053383, 41.8726501, -74.9832687, 75.4332504
2: -27.2823734, 45.4918365, -25.2556973, 42.0983925, -69.3807678, 70.7475128
3: -43.3220978, 54.6532707, -39.7605858, 50.6513634, -93.9734497, 94.4138565
4: -36.1003265, 60.1447449, -33.4442749, 55.6784706, -91.7787933, 93.5890198

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7044574, upper bound: 146.6975431
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7044302, upper bound: 146.6975260
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7047802, upper bound: 146.6982024
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.7565994, 70.2050858, -49.9627228, 65.0157700, -118.7723694, 120.1678009
1: -42.3193741, 57.6221886, -39.3158722, 53.5048180, -95.8241806, 96.9380646
2: -34.8279877, 58.4526215, -32.5230789, 53.5936279, -88.4216156, 90.9757004
3: -55.0447731, 69.8646393, -50.9778252, 64.7967606, -119.8414993, 120.8424530
4: -46.1298752, 77.5479279, -43.0311394, 71.2576828, -117.3875580, 120.5790710

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7042723, upper bound: 146.6924620
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6820617, upper bound: 146.6426459
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6820617, upper bound: 146.6916887
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.7565994, 70.2050858, -40.1203499, 50.7516823, -104.5082779, 110.3254318
1: -42.3193741, 57.6221886, -30.4053383, 41.8726501, -84.1920013, 88.0275269
2: -34.8279877, 58.4526215, -25.2556973, 42.0983925, -76.9263763, 83.7083130
3: -55.0447731, 69.8646393, -39.7605858, 50.6513634, -105.6961136, 109.6252213
4: -46.1298752, 77.5479279, -33.4442749, 55.6784706, -101.8083496, 110.9922028

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7042722, upper bound: 146.6969715
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7042644, upper bound: 146.6957464
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -56.3390350, 73.6385880, -116.2447052, 111.2400818
1: -33.1106224, 45.0279160, -44.6075058, 61.0470200, -94.1576385, 89.6353989
2: -27.2823734, 45.4918365, -36.9088554, 60.8001595, -88.0825348, 82.4006882
3: -43.3220978, 54.6532707, -57.8986092, 73.9244232, -117.2465210, 112.5518799
4: -36.1003265, 60.1447449, -48.8650551, 81.1451187, -117.2454376, 109.0097961

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7039950, upper bound: 146.6931420
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031056, upper bound: 146.6926598
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7030508, upper bound: 146.6895457
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -42.6061211, 54.9010429, -47.6786270, 61.6097183, -104.2158356, 102.5796661
1: -33.1106224, 45.0279160, -36.9107399, 50.8506050, -83.9612198, 81.9386215
2: -27.2823734, 45.4918365, -30.5371628, 51.3591270, -78.6415024, 76.0289841
3: -43.3220978, 54.6532707, -48.0740280, 61.5079536, -104.8300476, 102.7272949
4: -36.1003265, 60.1447449, -40.5163841, 68.1169739, -104.2173004, 100.6611328

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7039950, upper bound: 146.6981393
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031056, upper bound: 146.6968076
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7030508, upper bound: 146.6952127
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.7659836, 70.2163086, -56.3390350, 73.6385880, -127.4045486, 126.5553436
1: -42.3269310, 57.6318207, -44.6075058, 61.0470200, -103.3739471, 102.2393188
2: -34.8345337, 58.4619865, -36.9088554, 60.8001595, -95.6346893, 95.3708344
3: -55.0541573, 69.8761063, -57.8986092, 73.9244232, -128.9785767, 127.7747040
4: -46.1383362, 77.5608292, -48.8650551, 81.1451187, -127.2834549, 126.4258881

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038766, upper bound: 146.6919696
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038991, upper bound: 146.6919604
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.7659836, 70.2163086, -47.6786270, 61.6097183, -115.3757019, 117.8949356
1: -42.3269310, 57.6318207, -36.9107399, 50.8506050, -93.1775131, 94.5425415
2: -34.8345337, 58.4619865, -30.5371628, 51.3591270, -86.1936646, 88.9991302
3: -55.0541573, 69.8761063, -48.0740280, 61.5079536, -116.5621109, 117.9501266
4: -46.1383362, 77.5608292, -40.5163841, 68.1169739, -114.2553101, 118.0772095

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038766, upper bound: 146.6976059
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038991, upper bound: 146.6961491
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -54.8935204, 71.9935226, -54.1649742, 70.9842606, -125.8777771, 126.1584930
1: -43.7559738, 59.0566177, -43.1738892, 58.2160645, -101.9720383, 102.2305069
2: -35.9394417, 59.3323936, -35.4552650, 58.4927711, -94.4322128, 94.7876282
3: -56.7730713, 71.7635269, -56.0005150, 70.7430801, -127.5161438, 127.7640381
4: -47.5056877, 78.9019928, -46.8653336, 77.7591171, -125.2648010, 125.7673264

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6275681, upper bound: 146.5540884
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976774, upper bound: 146.6976774
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -54.8935204, 71.9935226, -42.6822357, 54.9827347, -109.8762512, 114.6757584
1: -43.7559738, 59.0566177, -33.1644592, 45.0954857, -88.8514557, 92.2210770
2: -35.9394417, 59.3323936, -27.3295212, 45.5579834, -81.4974213, 86.6619110
3: -56.7730713, 71.7635269, -43.3842506, 54.7341614, -111.5072327, 115.1477737
4: -47.5056877, 78.9019928, -36.1597328, 60.2308922, -107.7365799, 115.0617142

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6275681, upper bound: 146.5541013
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6976774, upper bound: 146.6977600
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -61.7200356, 81.1029663, -54.1649742, 70.9842606, -132.7042999, 135.2679443
1: -49.2398033, 66.7187576, -43.1738892, 58.2160645, -107.4558716, 109.8926468
2: -40.5049629, 66.9290161, -35.4552650, 58.4927711, -98.9977341, 102.3842468
3: -63.8531952, 81.0615082, -56.0005150, 70.7430801, -134.5962524, 137.0619965
4: -53.5745544, 89.2021790, -46.8653336, 77.7591171, -131.3336639, 136.0675049

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6503707, upper bound: 146.5863986
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6865776
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -61.7200356, 81.1029663, -42.6822357, 54.9827347, -116.7027740, 123.7852020
1: -49.2398033, 66.7187576, -33.1644592, 45.0954857, -94.3352890, 99.8832169
2: -40.5049629, 66.9290161, -27.3295212, 45.5579834, -86.0629425, 94.2585373
3: -63.8531952, 81.0615082, -43.3842506, 54.7341614, -118.5873566, 124.4457550
4: -53.5745544, 89.2021790, -36.1597328, 60.2308922, -113.8054428, 125.3618927

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6503707, upper bound: 146.5863986
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6947009, upper bound: 146.6866627
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -78.6598434, 104.7793884, -58.0480080, 76.0305405, -154.6903839, 162.8273926
1: -62.6864204, 86.3446045, -46.0885239, 62.4561768, -125.1425934, 132.4331207
2: -51.7383270, 87.3350754, -37.9866867, 62.7829514, -114.5212784, 125.3217621
3: -81.7546539, 104.3666382, -59.7441063, 75.8010406, -157.5556946, 164.1107178
4: -68.5973663, 116.7258606, -50.2364769, 83.5897522, -152.1871033, 166.9623108

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6768526, upper bound: 146.6333789
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754295, upper bound: 146.6333926
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6280297, upper bound: 146.5568120
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6769102, upper bound: 146.6333926
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -78.6598434, 104.7793884, -51.7050972, 67.3109055, -145.9707489, 156.4844666
1: -62.6864204, 86.3446045, -40.5460663, 55.2616768, -117.9480972, 126.8906631
2: -51.7383270, 87.3350754, -33.3965683, 56.0460510, -107.7843781, 120.7316437
3: -81.7546539, 104.3666382, -52.7403030, 67.0014725, -148.7561188, 157.1069183
4: -68.5973663, 116.7258606, -44.2198944, 74.2704697, -142.8678284, 160.9457397

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6768526, upper bound: 146.6334080
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6754295, upper bound: 146.6334217
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6280297, upper bound: 146.5568120
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6769102, upper bound: 146.6334217
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -64.9658508, 85.9501801, -60.6591187, 79.7301712, -144.6959686, 146.6092987
1: -51.9960442, 70.6328888, -48.4117584, 65.5827179, -117.5787582, 119.0446396
2: -42.7397270, 71.0999146, -39.8212395, 65.7914658, -108.5311737, 110.9211502
3: -67.6466370, 85.7477036, -62.7888260, 79.6712570, -147.3179016, 148.5365295
4: -56.5694389, 94.8905563, -52.6769295, 87.6756439, -144.2450867, 147.5674896

Time for backsubstitution: 1.60 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.50 + 417.68 = 421.17 seconds
