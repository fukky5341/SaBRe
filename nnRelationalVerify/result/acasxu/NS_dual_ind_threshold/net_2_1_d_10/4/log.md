## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 155.85206263506


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228)
1: (-62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323)
2: (-55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172)
3: (-72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134)
4: (-72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 2.30 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -155.8676494, upper bound: 155.8676494

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8584031, upper bound: 155.8630827
time: 0.92 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 1.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.11 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 4, lower bound: -155.8584031, upper bound: 155.8630827
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -60.2190781, 69.0278397, -81.3169098, 96.3277130, -156.5467834, 150.3447418
1: -46.0649567, 55.6417961, -62.8572426, 77.7227859, -123.7877350, 118.4990234
2: -40.2784386, 54.2545853, -55.0245743, 76.3968430, -116.6752777, 109.2791595
3: -52.6274223, 67.0956802, -72.9154510, 93.5727158, -146.2001190, 140.0111389
4: -52.6573296, 73.0029373, -72.2197571, 103.2277756, -155.8850861, 145.2226868

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 0.93 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 1.07 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -108.3899231, 133.7219238, -75.6169586, 89.3531570, -197.7430267, 209.3388824
1: -85.1367645, 107.0074463, -58.4705429, 72.1841736, -157.3209381, 165.4779968
2: -74.3936386, 106.0023499, -51.1664467, 70.8636856, -145.2573090, 157.1687927
3: -99.2187576, 129.2454376, -67.6475906, 87.0211868, -186.2399445, 196.8930359
4: -98.1038055, 142.7249908, -67.1893158, 95.6991959, -193.8029938, 209.9143066

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 1.00 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 0.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -60.2190781, 69.0278397, -60.2190781, 69.0278397, -129.2469025, 129.2469025
1: -46.0649567, 55.6417961, -46.0649567, 55.6417961, -101.7067337, 101.7067337
2: -40.2784386, 54.2545853, -40.2784386, 54.2545853, -94.5329895, 94.5329895
3: -52.6274223, 67.0956802, -52.6274223, 67.0956802, -119.7230988, 119.7230988
4: -52.6573296, 73.0029373, -52.6573296, 73.0029373, -125.6602631, 125.6602631

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8526818, upper bound: 155.8528078
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8581304, upper bound: 155.8617025
time: 1.05 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -60.2190781, 69.0278397, -108.3899231, 133.7219238, -193.9409943, 177.4177246
1: -46.0649567, 55.6417961, -85.1367645, 107.0074463, -153.0724030, 140.7785645
2: -40.2784386, 54.2545853, -74.3936386, 106.0023499, -146.2807922, 128.6481781
3: -52.6274223, 67.0956802, -99.2187576, 129.2454376, -181.8728638, 166.3144379
4: -52.6573296, 73.0029373, -98.1038055, 142.7249908, -195.3823090, 171.1067505

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8526818, upper bound: 155.8528078
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8526818, upper bound: 155.8617025
time: 0.86 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -108.3899231, 133.7219238, -60.2190781, 69.0278397, -177.4177246, 193.9409943
1: -85.1367645, 107.0074463, -46.0649567, 55.6417961, -140.7785492, 153.0724030
2: -74.3936386, 106.0023499, -40.2784386, 54.2545853, -128.6481934, 146.2807922
3: -99.2187576, 129.2454376, -52.6274223, 67.0956802, -166.3144379, 181.8728638
4: -98.1038055, 142.7249908, -52.6573296, 73.0029373, -171.1067505, 195.3823090

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8389050
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567250, upper bound: 155.8567250
time: 0.89 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -108.3899231, 133.7219238, -108.3899231, 133.7219238, -242.1118164, 242.1118164
1: -85.1367645, 107.0074463, -85.1367645, 107.0074463, -192.1442108, 192.1442108
2: -74.3936386, 106.0023499, -74.3936386, 106.0023499, -180.3959961, 180.3959961
3: -99.2187576, 129.2454376, -99.2187576, 129.2454376, -228.4642029, 228.4642029
4: -98.1038055, 142.7249908, -98.1038055, 142.7249908, -240.8287964, 240.8287964

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8389050
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8567250
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.35 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8526818, upper bound: 155.8528078
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8581304, upper bound: 155.8617025
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8526818, upper bound: 155.8528078
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8526818, upper bound: 155.8617025
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8389050
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8567250, upper bound: 155.8567250
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8389050
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8567250

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -40.9217491, 46.0795670, -58.0577965, 66.3627396, -107.2844849, 104.1373596
1: -31.2777653, 37.2222519, -44.3639908, 53.4697990, -84.7475433, 81.5862427
2: -27.3006096, 36.1713181, -38.7909164, 52.1195335, -79.4201431, 74.9622116
3: -35.5017929, 44.8549156, -50.6296730, 64.4623184, -99.9641113, 95.4845886
4: -35.6733818, 48.6733131, -50.6860695, 70.1152420, -105.7886200, 99.3593826

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613185, upper bound: 155.8613185
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613185, upper bound: 155.8613431
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -59.8527756, 68.5774918, -126.5137482, 126.0706177
1: -44.3237762, 53.3550873, -45.7839432, 55.2742844, -99.5980606, 99.1390305
2: -38.7481384, 51.9732895, -40.0318985, 53.8888206, -92.6369553, 92.0051880
3: -50.5645981, 64.3627167, -52.2944145, 66.6559219, -117.2205200, 116.6571274
4: -50.6411171, 69.9437561, -52.3323059, 72.5115967, -123.1527100, 122.2760620

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634050
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634296
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -40.9217491, 46.0795670, -105.7787018, 130.1714935, -171.0932465, 151.8582153
1: -31.2777653, 37.2222519, -83.0240326, 104.1414490, -135.4192200, 120.2462845
2: -27.3006096, 36.1713181, -72.5307236, 103.1036530, -130.4042358, 108.7020416
3: -35.5017929, 44.8549156, -96.6085434, 125.8380051, -161.3397675, 141.4634552
4: -35.6733818, 48.6733131, -95.6322021, 138.7948914, -174.4682465, 144.3055115

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8500387
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8528078
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -108.0825806, 133.3024902, -191.2387238, 174.3004303
1: -44.3237762, 53.3550873, -84.8943481, 106.6633301, -150.9871063, 138.2494354
2: -38.7481384, 51.9732895, -74.1782150, 105.6524048, -144.4005280, 126.1515045
3: -50.5645981, 64.3627167, -98.9163666, 128.8399963, -179.4045563, 163.2790527
4: -50.6411171, 69.9437561, -97.8169022, 142.2529449, -192.8940582, 167.7606506

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8500387
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8617025
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -106.1845322, 130.7253113, -59.8527756, 68.5774918, -174.7619781, 190.5780792
1: -83.3945312, 104.5468597, -45.7839432, 55.2742844, -138.6688232, 150.3308105
2: -72.8464279, 103.5035629, -40.0318985, 53.8888206, -126.7352371, 143.5354614
3: -97.0476532, 126.3464890, -52.2944145, 66.6559219, -163.7035828, 178.6408997
4: -96.0440063, 139.3540802, -52.3323059, 72.5115967, -168.5555725, 191.6863708

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8526818
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8581304
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -106.1845322, 130.7253113, -108.0825806, 133.3024902, -239.4869995, 238.8078918
1: -83.3945312, 104.5468597, -84.8943481, 106.6633301, -190.0578613, 189.4411926
2: -72.8464279, 103.5035629, -74.1782150, 105.6524048, -178.4988251, 177.6817780
3: -97.0476532, 126.3464890, -98.9163666, 128.8399963, -225.8876190, 225.2628174
4: -96.0440063, 139.3540802, -97.8169022, 142.2529449, -238.2969360, 237.1709747

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8389050, upper bound: 155.8470451
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8389050, upper bound: 155.8567250
time: 1.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.32 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8613185, upper bound: 155.8613185
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8613185, upper bound: 155.8613431
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634050
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634296
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8500387
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8528078
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8500387
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8617025
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8526818
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8581304
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8389050, upper bound: 155.8470451
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -155.8389050, upper bound: 155.8567250

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.9217491, 46.0795670, -40.9217491, 46.0795670, -87.0013123, 87.0013123
1: -31.2777653, 37.2222519, -31.2777653, 37.2222519, -68.5000153, 68.5000153
2: -27.3006096, 36.1713181, -27.3006096, 36.1713181, -63.4719276, 63.4719276
3: -35.5017929, 44.8549156, -35.5017929, 44.8549156, -80.3567047, 80.3567047
4: -35.6733818, 48.6733131, -35.6733818, 48.6733131, -84.3466873, 84.3466873

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8459015, upper bound: 155.8545273
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436033
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.9217491, 46.0795670, -57.9362602, 66.2178497, -107.1395950, 104.0158234
1: -31.2777653, 37.2222519, -44.3237762, 53.3550873, -84.6328506, 81.5460281
2: -27.3006096, 36.1713181, -38.7481384, 51.9732895, -79.2738953, 74.9194489
3: -35.5017929, 44.8549156, -50.5645981, 64.3627167, -99.8645096, 95.4195023
4: -35.6733818, 48.6733131, -50.6411171, 69.9437561, -105.6171417, 99.3144226

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8459015, upper bound: 155.8547893
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436034
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -40.9217491, 46.0795670, -104.0158234, 107.1395874
1: -44.3237762, 53.3550873, -31.2777653, 37.2222519, -81.5460281, 84.6328430
2: -38.7481384, 51.9732895, -27.3006096, 36.1713181, -74.9194489, 79.2738953
3: -50.5645981, 64.3627167, -35.5017929, 44.8549156, -95.4195023, 99.8645096
4: -50.6411171, 69.9437561, -35.6733818, 48.6733131, -99.3144226, 105.6171417

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608217, upper bound: 155.8629334
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608391, upper bound: 155.8616230
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -57.9362602, 66.2178497, -124.1541061, 124.1541138
1: -44.3237762, 53.3550873, -44.3237762, 53.3550873, -97.6788635, 97.6788635
2: -38.7481384, 51.9732895, -38.7481384, 51.9732895, -90.7214279, 90.7214279
3: -50.5645981, 64.3627167, -50.5645981, 64.3627167, -114.9273071, 114.9273071
4: -50.6411171, 69.9437561, -50.6411171, 69.9437561, -120.5848694, 120.5848694

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608217, upper bound: 155.8629566
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608391, upper bound: 155.8616253
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -40.9217491, 46.0795670, -106.1845322, 130.7253113, -171.6470490, 152.2640381
1: -31.2777653, 37.2222519, -83.3945312, 104.5468597, -135.8246307, 120.6167831
2: -27.3006096, 36.1713181, -72.8464279, 103.5035629, -130.8041687, 109.0177383
3: -35.5017929, 44.8549156, -97.0476532, 126.3464890, -161.8482361, 141.9025421
4: -35.6733818, 48.6733131, -96.0440063, 139.3540802, -175.0274353, 144.7172852

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8401475, upper bound: 155.8528078
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8384147, upper bound: 155.8435508
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -106.1845322, 130.7253113, -188.6615448, 172.4023285
1: -44.3237762, 53.3550873, -83.3945312, 104.5468597, -148.8706360, 136.7496185
2: -38.7481384, 51.9732895, -72.8464279, 103.5035629, -142.2517090, 124.8197174
3: -50.5645981, 64.3627167, -97.0476532, 126.3464890, -176.9110565, 161.4103546
4: -50.6411171, 69.9437561, -96.0440063, 139.3540802, -189.9951935, 165.9877625

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8556981
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8383486, upper bound: 155.8445434
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -106.1845322, 130.7253113, -40.9217491, 46.0795670, -152.2640381, 171.6470490
1: -83.3945312, 104.5468597, -31.2777653, 37.2222519, -120.6167831, 135.8246307
2: -72.8464279, 103.5035629, -27.3006096, 36.1713181, -109.0177383, 130.8041687
3: -97.0476532, 126.3464890, -35.5017929, 44.8549156, -141.9025421, 161.8482361
4: -96.0440063, 139.3540802, -35.6733818, 48.6733131, -144.7172699, 175.0274353

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8444229, upper bound: 155.8500066
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8435508, upper bound: 155.8465010
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -106.1845322, 130.7253113, -57.9362602, 66.2178497, -172.4023285, 188.6615448
1: -83.3945312, 104.5468597, -44.3237762, 53.3550873, -136.7496185, 148.8706360
2: -72.8464279, 103.5035629, -38.7481384, 51.9732895, -124.8197174, 142.2517090
3: -97.0476532, 126.3464890, -50.5645981, 64.3627167, -161.4103546, 176.9110565
4: -96.0440063, 139.3540802, -50.6411171, 69.9437561, -165.9877625, 189.9951935

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8444229, upper bound: 155.8510080
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8435508, upper bound: 155.8465062
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -106.1845322, 130.7253113, -106.1845322, 130.7253113, -236.9098053, 236.9098053
1: -83.3945312, 104.5468597, -83.3945312, 104.5468597, -187.9413910, 187.9413910
2: -72.8464279, 103.5035629, -72.8464279, 103.5035629, -176.3499908, 176.3499908
3: -97.0476532, 126.3464890, -97.0476532, 126.3464890, -223.3941040, 223.3941040
4: -96.0440063, 139.3540802, -96.0440063, 139.3540802, -235.3980713, 235.3980713

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8389050, upper bound: 155.8511050
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8382961, upper bound: 155.8464537
time: 1.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.73 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8459015, upper bound: 155.8545273
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436033
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8459015, upper bound: 155.8547893
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436034
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8608217, upper bound: 155.8629334
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8608391, upper bound: 155.8616230
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8608217, upper bound: 155.8629566
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8608391, upper bound: 155.8616253
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8401475, upper bound: 155.8528078
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8384147, upper bound: 155.8435508
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8556981
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8383486, upper bound: 155.8445434
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8444229, upper bound: 155.8500066
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8435508, upper bound: 155.8465010
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8444229, upper bound: 155.8510080
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8435508, upper bound: 155.8465062
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8389050, upper bound: 155.8511050
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 4, lower bound: -155.8382961, upper bound: 155.8464537

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -35.7231407, 39.7865067, -40.9217491, 46.0795670, -81.8027039, 80.7082520
1: -27.2421684, 32.1181641, -31.2777653, 37.2222519, -64.4644165, 63.3959274
2: -23.7660103, 31.1537247, -27.3006096, 36.1713181, -59.9373283, 58.4543266
3: -30.8049335, 38.6498909, -35.5017929, 44.8549156, -75.6598511, 74.1516876
4: -31.0086708, 41.9031792, -35.6733818, 48.6733131, -79.6819687, 77.5765610

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436695
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436695
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -35.7231407, 39.7865067, -57.9362602, 66.2178497, -101.9409790, 97.7227631
1: -27.2421684, 32.1181641, -44.3237762, 53.3550873, -80.5972366, 76.4419403
2: -23.7660103, 31.1537247, -38.7481384, 51.9732895, -75.7392883, 69.9018555
3: -30.8049335, 38.6498909, -50.5645981, 64.3627167, -95.1676483, 89.2144775
4: -31.0086708, 41.9031792, -50.6411171, 69.9437561, -100.9524231, 92.5442963

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8446626, upper bound: 155.8436034
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8446626, upper bound: 155.8436034
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -65.8193665, 75.0918884, -38.9490471, 43.7663155, -109.5856705, 114.0409393
1: -50.3528633, 60.4418297, -29.7718430, 35.3651276, -85.7179871, 90.2136688
2: -44.0209045, 58.9657593, -25.9741955, 34.3508835, -78.3717880, 84.9399567
3: -57.4032402, 73.0119781, -33.7795258, 42.6122322, -100.0154724, 106.7915039
4: -57.4745331, 79.4589767, -33.9406586, 46.2365761, -103.7111053, 113.3996353

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607994, upper bound: 155.8629334
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607938, upper bound: 155.8629254
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -54.9395752, 62.5228233, -40.8546448, 45.9995003, -100.9390717, 103.3774719
1: -41.9862709, 50.3208694, -31.2261620, 37.1570206, -79.1432724, 81.5470276
2: -36.6859436, 49.0053902, -27.2551575, 36.1076660, -72.7936096, 76.2605438
3: -47.8192101, 60.6894035, -35.4418602, 44.7755699, -92.5947723, 96.1312637
4: -47.9238625, 65.9192276, -35.6135559, 48.5867653, -96.5106125, 101.5327759

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8544319, upper bound: 155.8451532
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8427054, upper bound: 155.8431305
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -65.8193665, 75.0918884, -54.3523140, 62.0487137, -127.8680801, 129.4441986
1: -50.3528633, 60.4418297, -41.6142311, 50.0248985, -100.3777466, 102.0560608
2: -44.0209045, 58.9657593, -36.3664246, 48.7055168, -92.7264023, 95.3321838
3: -57.4032402, 73.0119781, -47.4511833, 60.3592567, -117.7624969, 120.4631653
4: -57.4745331, 79.4589767, -47.5348854, 65.5680084, -123.0425415, 126.9938660

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608985, upper bound: 155.8609064
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608985, upper bound: 155.8609239
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -54.9395752, 62.5228233, -57.8517036, 66.1119843, -121.0515594, 120.3745270
1: -41.9862709, 50.3208694, -44.2574043, 53.2681923, -95.2544556, 94.5782700
2: -36.6859436, 49.0053902, -38.6895828, 51.8879547, -88.5738983, 87.6949692
3: -47.8192101, 60.6894035, -50.4860764, 64.2578278, -112.0770416, 111.1754761
4: -47.9238625, 65.9192276, -50.5638657, 69.8282242, -117.7520828, 116.4830856

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8534972, upper bound: 155.8452763
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8429990, upper bound: 155.8430486
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -35.7231407, 39.7865067, -106.1845322, 130.7253113, -166.4484406, 145.9710236
1: -27.2421684, 32.1181641, -83.3945312, 104.5468597, -131.7890167, 115.5126953
2: -23.7660103, 31.1537247, -72.8464279, 103.5035629, -127.2695618, 104.0001373
3: -30.8049335, 38.6498909, -97.0476532, 126.3464890, -157.1513824, 135.6975403
4: -31.0086708, 41.9031792, -96.0440063, 139.3540802, -170.3627472, 137.9471741

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8465010, upper bound: 155.8435508
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8465010, upper bound: 155.8435508
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -51.6664810, 58.6859055, -106.1845322, 130.7253113, -182.3917847, 164.8704224
1: -39.4619751, 47.2641678, -83.3945312, 104.5468597, -144.0088348, 130.6586914
2: -34.4859314, 45.9825592, -72.8464279, 103.5035629, -137.9895020, 118.8289871
3: -44.9092712, 56.9975624, -97.0476532, 126.3464890, -171.2557678, 154.0452118
4: -45.0333710, 61.8690224, -96.0440063, 139.3540802, -184.3874512, 157.9129944

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434
time: 3.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.19 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436695
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8436695, upper bound: 155.8436695
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8446626, upper bound: 155.8436034
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8446626, upper bound: 155.8436034
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8607994, upper bound: 155.8629334
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8607938, upper bound: 155.8629254
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8544319, upper bound: 155.8451532
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8427054, upper bound: 155.8431305
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8608985, upper bound: 155.8609064
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8608985, upper bound: 155.8609239
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8534972, upper bound: 155.8452763
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8429990, upper bound: 155.8430486
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8465010, upper bound: 155.8435508
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8465010, upper bound: 155.8435508
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.19
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -65.8193665, 75.0918884, -36.4976463, 40.9924545, -106.8117981, 111.5895386
1: -50.3528633, 60.4418297, -27.9415531, 33.1709557, -83.5238190, 88.3833847
2: -44.0209045, 58.9657593, -24.3520775, 32.2601433, -76.2810440, 83.3178406
3: -57.4032402, 73.0119781, -31.7381859, 40.0362244, -97.4394684, 104.7501678
4: -57.4745331, 79.4589767, -31.8598652, 43.5597572, -101.0342865, 111.3188400

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607936, upper bound: 155.8629334
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607994, upper bound: 155.8623582
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -65.8193665, 75.0918884, -37.8068466, 42.4556656, -108.2750320, 112.8987274
1: -50.3528633, 60.4418297, -28.9005527, 34.3118286, -84.6646881, 89.3423843
2: -44.0209045, 58.9657593, -25.2122860, 33.3278160, -77.3487091, 84.1780472
3: -57.4032402, 73.0119781, -32.7883720, 41.3344383, -98.7376785, 105.8003540
4: -57.4745331, 79.4589767, -32.9454727, 44.8700218, -102.3445511, 112.4044495

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607880, upper bound: 155.8629254
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607938, upper bound: 155.8623557
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -54.9395752, 62.5228233, -35.6627121, 39.7142563, -94.6538315, 98.1855316
1: -41.9862709, 50.3208694, -27.1957741, 32.0591736, -74.0454178, 77.5166473
2: -36.6859436, 49.0053902, -23.7251129, 31.0961609, -67.7821045, 72.7304993
3: -47.8192101, 60.6894035, -30.7510910, 38.5781517, -86.3973618, 91.4404907
4: -47.9238625, 65.9192276, -30.9547997, 41.8246574, -89.7484970, 96.8740234

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8427054, upper bound: 155.8431305
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8427054, upper bound: 155.8431305
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -65.8193665, 75.0918884, -65.8193665, 75.0918884, -140.9112549, 140.9112549
1: -50.3528633, 60.4418297, -50.3528633, 60.4418297, -110.7946930, 110.7946930
2: -44.0209045, 58.9657593, -44.0209045, 58.9657593, -102.9866638, 102.9866638
3: -57.4032402, 73.0119781, -57.4032402, 73.0119781, -130.4152222, 130.4152222
4: -57.4745331, 79.4589767, -57.4745331, 79.4589767, -136.9335022, 136.9335022

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8609045, upper bound: 155.8629168
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608842, upper bound: 155.8629168
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -65.8193665, 75.0918884, -54.5213394, 62.0201912, -127.8395538, 129.6132202
1: -50.3528633, 60.4418297, -41.6670227, 49.9240227, -100.2768860, 102.1088562
2: -44.0209045, 58.9657593, -36.4041405, 48.6123657, -92.6332550, 95.3699036
3: -57.4032402, 73.0119781, -47.4489594, 60.2118301, -117.6150513, 120.4609375
4: -57.4745331, 79.4589767, -47.5546532, 65.3949051, -122.8694305, 127.0136261

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8587635, upper bound: 155.8562485
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8590082, upper bound: 155.8614256
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -54.9395752, 62.5228233, -51.5861282, 58.5838318, -113.5234070, 114.1089478
1: -41.9862709, 50.3208694, -39.3984795, 47.1806526, -89.1669006, 89.7193451
2: -36.6859436, 49.0053902, -34.4299622, 45.9001541, -82.5860977, 83.4353485
3: -47.8192101, 60.6894035, -44.8340073, 56.8964729, -104.7156830, 105.5234070
4: -47.9238625, 65.9192276, -44.9594460, 61.7573814, -109.6812363, 110.8786621

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8429990, upper bound: 155.8430486
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8531632, upper bound: 155.8449367
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8543853, upper bound: 155.8451929
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8543659, upper bound: 155.8451478
time: 1.15 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 7.65 seconds
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8607936, upper bound: 155.8629334
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8607994, upper bound: 155.8623582
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8607880, upper bound: 155.8629254
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8607938, upper bound: 155.8623557
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8427054, upper bound: 155.8431305
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8427054, upper bound: 155.8431305
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8609045, upper bound: 155.8629168
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8608842, upper bound: 155.8629168
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8587635, upper bound: 155.8562485
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8590082, upper bound: 155.8614256
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8543853, upper bound: 155.8451929
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.65
Output dim: 4, lower bound: -155.8543659, upper bound: 155.8451478

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -83.5034714, 93.9685516, -35.2149048, 39.5237694, -123.0272369, 129.1834412
1: -63.7773132, 75.7958374, -26.9623337, 31.9750080, -95.7523041, 102.7581711
2: -55.6933517, 73.8310776, -23.4897404, 31.0951500, -86.7884979, 97.3208160
3: -72.2092896, 91.5789108, -30.6198921, 38.5897636, -110.7990570, 122.1987915
4: -72.6306992, 99.4831772, -30.7343960, 41.9983139, -114.6289978, 130.2175751

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -63.3688469, 72.2695389, -36.4976463, 40.9924545, -104.3612823, 108.7671814
1: -48.5502510, 58.1747551, -27.9415531, 33.1709557, -81.7212067, 86.1163101
2: -42.4419403, 56.7468987, -24.3520775, 32.2601433, -74.7020874, 81.0989761
3: -55.3382950, 70.2878036, -31.7381859, 40.0362244, -95.3745193, 102.0259857
4: -55.4249573, 76.5109787, -31.8598652, 43.5597572, -98.9847107, 108.3708420

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565501, upper bound: 155.8519520
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607319
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607031, upper bound: 155.8607630
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -83.5034714, 93.9685516, -36.3425522, 40.7926521, -124.2961273, 130.3110962
1: -63.7773132, 75.7958374, -27.7842960, 32.9628639, -96.7401733, 103.5801315
2: -55.6933517, 73.8310776, -24.2331161, 32.0173874, -87.7107315, 98.0641861
3: -72.2092896, 91.5789108, -31.5277042, 39.7026062, -111.9118958, 123.1066132
4: -72.6306992, 99.4831772, -31.6695042, 43.1187401, -115.7494278, 131.1526794

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586293, upper bound: 155.8614610
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607083, upper bound: 155.8628783
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8596553, upper bound: 155.8627454
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -63.3688469, 72.2695389, -37.8068466, 42.4556656, -105.8245087, 110.0763855
1: -48.5502510, 58.1747551, -28.9005527, 34.3118286, -82.8620758, 87.0753098
2: -42.4419403, 56.7468987, -25.2122860, 33.3278160, -75.7697601, 81.9591827
3: -55.3382950, 70.2878036, -32.7883720, 41.3344383, -96.6727295, 103.0761719
4: -55.4249573, 76.5109787, -32.9454727, 44.8700218, -100.2949829, 109.4564362

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8614610
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8623557
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -63.9723282, 72.7513428, -65.8193665, 75.0918884, -139.0642090, 138.5707092
1: -48.9926109, 58.6176910, -50.3528633, 60.4418297, -109.4344406, 108.9705505
2: -42.8426323, 57.1732521, -44.0209045, 58.9657593, -101.8083954, 101.1941452
3: -55.8395424, 70.8680420, -57.4032402, 73.0119781, -128.8515167, 128.2712708
4: -55.9672127, 77.1005936, -57.4745331, 79.4589767, -135.4261932, 134.5751343

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -64.4276733, 73.4758987, -65.8193665, 75.0918884, -139.5195465, 139.2952576
1: -49.2981796, 59.1440315, -50.3528633, 60.4418297, -109.7400055, 109.4968948
2: -43.0962944, 57.6944580, -44.0209045, 58.9657593, -102.0620499, 101.7153625
3: -56.1992683, 71.4468384, -57.4032402, 73.0119781, -129.2112427, 128.8500519
4: -56.2723427, 77.7576904, -57.4745331, 79.4589767, -135.7313080, 135.2322235

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -57.7400208, 65.3608322, -50.7541046, 57.7583656, -115.4983826, 116.1149216
1: -43.9277534, 52.5258560, -38.7976723, 46.4651642, -90.3929138, 91.3235245
2: -38.4120255, 51.2234726, -33.8960953, 45.2496910, -83.6617126, 85.1195679
3: -49.9438705, 63.4120712, -44.1872520, 56.0209732, -105.9648438, 107.5993195
4: -50.0942535, 68.8373566, -44.2813301, 60.8605537, -110.9548035, 113.1186829

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562266, upper bound: 155.8561094
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562266, upper bound: 155.8562484
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -62.2001839, 71.0469208, -54.5213394, 62.0201912, -124.2203598, 125.5682602
1: -47.6223106, 57.1790695, -41.6670227, 49.9240227, -97.5463333, 98.8460922
2: -41.6193466, 55.7893486, -36.4041405, 48.6123657, -90.2317123, 92.1934891
3: -54.3366394, 69.0618134, -47.4489594, 60.2118301, -114.5484467, 116.5107727
4: -54.3550720, 75.1997299, -47.5546532, 65.3949051, -119.7499619, 122.7543793

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564254, upper bound: 155.8608594
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8559399, upper bound: 155.8614256
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -53.6355133, 60.9454651, -51.5861282, 58.5838318, -112.2193451, 112.5315933
1: -40.9800911, 49.0546913, -39.3984795, 47.1806526, -88.1607437, 88.4531708
2: -35.8027115, 47.7555313, -34.4299622, 45.9001541, -81.7028656, 82.1854935
3: -46.6380539, 59.1684151, -44.8340073, 56.8964729, -103.5345306, 104.0024033
4: -46.7681732, 64.2361984, -44.9594460, 61.7573814, -108.5255585, 109.1956329

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8496175, upper bound: 155.8429699
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8493770, upper bound: 155.8429403
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -54.3273621, 61.7940445, -51.5861282, 58.5838318, -112.9111862, 113.3801727
1: -41.5190048, 49.7360916, -39.3984795, 47.1806526, -88.6996460, 89.1345596
2: -36.2772064, 48.4280701, -34.4299622, 45.9001541, -82.1773605, 82.8580246
3: -47.2770920, 59.9833488, -44.8340073, 56.8964729, -104.1735687, 104.8173294
4: -47.3884583, 65.1432343, -44.9594460, 61.7573814, -109.1458435, 110.1026764

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8496176, upper bound: 155.8429699
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8493662, upper bound: 155.8429403
time: 1.03 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.62 seconds
NS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
NS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
NS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607319
NS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8607031, upper bound: 155.8607630
NS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8607083, upper bound: 155.8628783
NS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8596553, upper bound: 155.8627454
NS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8614610
NS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8623557
NS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
NS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
NS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
NS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8627645, upper bound: 155.8629169
NS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8562266, upper bound: 155.8561094
NS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8562266, upper bound: 155.8562484
NS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8564254, upper bound: 155.8608594
NS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8559399, upper bound: 155.8614256
NS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8496175, upper bound: 155.8429699
NS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8493770, upper bound: 155.8429403
NS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8496176, upper bound: 155.8429699
NS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 4, lower bound: -155.8493662, upper bound: 155.8429403

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.5034714, 93.9685516, -34.9739075, 39.2580719, -122.7615356, 128.9424438
1: -63.7773132, 75.7958374, -26.7778168, 31.7589359, -95.5362473, 102.5736542
2: -55.6933517, 73.8310776, -23.3282013, 30.8873501, -86.5807037, 97.1592712
3: -72.2092896, 91.5789108, -30.4137516, 38.3277054, -110.5369949, 121.9926605
4: -72.6306992, 99.4831772, -30.5236435, 41.7227249, -114.3534241, 130.0068207

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -83.5034714, 93.9685516, -35.1993599, 39.5222130, -123.0256805, 129.1678925
1: -63.7773132, 75.7958374, -26.9524441, 31.9716854, -95.7489929, 102.7482834
2: -55.6933517, 73.8310776, -23.4787922, 31.0990868, -86.7924347, 97.3098679
3: -72.2092896, 91.5789108, -30.6249485, 38.5937233, -110.8030090, 122.2038574
4: -72.6306992, 99.4831772, -30.7226028, 42.0155640, -114.6462631, 130.2057800

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -63.3688469, 72.2695389, -36.2489128, 40.7121582, -104.0810013, 108.5184479
1: -48.5502510, 58.1747551, -27.7490921, 32.9438286, -81.4940720, 85.9238434
2: -42.4419403, 56.7468987, -24.1843834, 32.0408592, -74.4828033, 80.9312820
3: -55.3382950, 70.2878036, -31.5226898, 39.7600212, -95.0983124, 101.8104935
4: -55.4249573, 76.5109787, -31.6403008, 43.2684479, -98.6934052, 108.1512756

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607319
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607319
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -63.3688469, 72.2695389, -36.5033760, 41.0126038, -104.3814392, 108.7728958
1: -48.5502510, 58.1747551, -27.9466991, 33.1850510, -81.7353058, 86.1214447
2: -42.4419403, 56.7468987, -24.3541794, 32.2812424, -74.7231827, 81.1010742
3: -55.3382950, 70.2878036, -31.7612324, 40.0626411, -95.4009399, 102.0490341
4: -55.4249573, 76.5109787, -31.8650322, 43.6010666, -99.0260239, 108.3760071

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607031, upper bound: 155.8607630
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607031, upper bound: 155.8607630
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -83.5034714, 93.9685516, -35.3288994, 39.6333694, -123.1368408, 129.2974396
1: -63.7773132, 75.7958374, -27.0226707, 32.0291748, -95.8064804, 102.8185120
2: -55.6933517, 73.8310776, -23.5637703, 31.1117859, -86.8051376, 97.3948517
3: -72.2092896, 91.5789108, -30.6463623, 38.5703201, -110.7796097, 122.2252579
4: -72.6306992, 99.4831772, -30.7967663, 41.9001198, -114.5308228, 130.2799377

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607083, upper bound: 155.8628783
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607083, upper bound: 155.8628783
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -83.5034714, 93.9685516, -35.9117203, 40.2799911, -123.7834625, 129.8802643
1: -63.7773132, 75.7958374, -27.4555321, 32.5475578, -96.3248596, 103.2513733
2: -55.6933517, 73.8310776, -23.9453163, 31.6098404, -87.3031921, 97.7763977
3: -72.2092896, 91.5789108, -31.1463699, 39.1984406, -111.4077301, 122.7252808
4: -72.6306992, 99.4831772, -31.2909832, 42.5710640, -115.2017670, 130.7741547

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8596553, upper bound: 155.8627454
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8596553, upper bound: 155.8627454
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -63.3688469, 72.2695389, -53.5605583, 59.2485504, -122.6173859, 125.8300934
1: -48.5502510, 58.1747551, -40.7351456, 48.0194244, -96.5696716, 98.9098892
2: -42.4419403, 56.7468987, -35.4819527, 46.5983582, -89.0402985, 92.2288513
3: -55.3382950, 70.2878036, -45.8934479, 57.9912910, -113.3295898, 116.1812515
4: -55.4249573, 76.5109787, -46.2466049, 62.8137169, -118.2386780, 122.7575836

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8605191
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8614610
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -63.3688469, 72.2695389, -34.4380074, 38.9053993, -102.2742462, 106.7075500
1: -48.5502510, 58.1747551, -26.4724121, 31.4414673, -79.9917145, 84.6471710
2: -42.4419403, 56.7468987, -23.0566540, 30.5704288, -73.0123672, 79.8035507
3: -55.3382950, 70.2878036, -30.0729237, 37.8880806, -93.2263794, 100.3607254
4: -55.4249573, 76.5109787, -30.1662598, 41.2300339, -96.6549911, 106.6772156

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8605191
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8614661
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -63.9723282, 72.7513428, -63.9723282, 72.7513428, -136.7236633, 136.7236633
1: -48.9926109, 58.6176910, -48.9926109, 58.6176910, -107.6103058, 107.6103058
2: -42.8426323, 57.1732521, -42.8426323, 57.1732521, -100.0158844, 100.0158844
3: -55.8395424, 70.8680420, -55.8395424, 70.8680420, -126.7075806, 126.7075806
4: -55.9672127, 77.1005936, -55.9672127, 77.1005936, -133.0678101, 133.0678101

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8624956, upper bound: 155.8620877
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -63.9723282, 72.7513428, -64.4276733, 73.4758987, -137.4482269, 137.1790161
1: -48.9926109, 58.6176910, -49.2981796, 59.1440315, -108.1366425, 107.9158707
2: -42.8426323, 57.1732521, -43.0962944, 57.6944580, -100.5370941, 100.2695312
3: -55.8395424, 70.8680420, -56.1992683, 71.4468384, -127.2863770, 127.0673065
4: -55.9672127, 77.1005936, -56.2723427, 77.7576904, -133.7248993, 133.3729248

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8624956, upper bound: 155.8620877
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -64.4276733, 73.4758987, -63.9723282, 72.7513428, -137.1790161, 137.4482269
1: -49.2981796, 59.1440315, -48.9926109, 58.6176910, -107.9158707, 108.1366425
2: -43.0962944, 57.6944580, -42.8426323, 57.1732521, -100.2695312, 100.5370941
3: -56.1992683, 71.4468384, -55.8395424, 70.8680420, -127.0673065, 127.2863770
4: -56.2723427, 77.7576904, -55.9672127, 77.1005936, -133.3729248, 133.7248993

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8624945, upper bound: 155.8620877
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -64.4276733, 73.4758987, -64.4276733, 73.4758987, -137.9035645, 137.9035645
1: -49.2981796, 59.1440315, -49.2981796, 59.1440315, -108.4422150, 108.4422150
2: -43.0962944, 57.6944580, -43.0962944, 57.6944580, -100.7907410, 100.7907410
3: -56.1992683, 71.4468384, -56.1992683, 71.4468384, -127.6461029, 127.6461029
4: -56.2723427, 77.7576904, -56.2723427, 77.7576904, -134.0300293, 134.0300140

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8624945, upper bound: 155.8620877
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -57.7400208, 65.3608322, -47.8892212, 53.9669075, -111.7069244, 113.2500305
1: -43.9277534, 52.5258560, -36.3983994, 43.3684845, -87.2962341, 88.9242554
2: -38.4120255, 51.2234726, -31.8371048, 42.1720200, -80.5840454, 83.0605774
3: -49.9438705, 63.4120712, -41.2748108, 52.2073288, -102.1511993, 104.6868820
4: -50.0942535, 68.8373566, -41.5129852, 56.5219460, -106.6161957, 110.3503418

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8559851, upper bound: 155.8559265
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8555875, upper bound: 155.8559278
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -57.7400208, 65.3608322, -51.5600166, 58.6148796, -116.3548965, 116.9208374
1: -43.9277534, 52.5258560, -39.4048767, 47.1900711, -91.1178207, 91.9307327
2: -38.4120255, 51.2234726, -34.4266205, 45.9367447, -84.3487701, 85.6500931
3: -49.9438705, 63.4120712, -44.8990669, 56.8835030, -106.8273773, 108.3111420
4: -50.0942535, 68.8373566, -44.9741592, 61.7972603, -111.8915100, 113.8115158

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8559851, upper bound: 155.8560566
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8560763, upper bound: 155.8560580
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -62.2001839, 71.0469208, -47.8892212, 53.9669075, -116.1670914, 118.9361420
1: -47.6223106, 57.1790695, -36.3983994, 43.3684845, -90.9907990, 93.5774689
2: -41.6193466, 55.7893486, -31.8371048, 42.1720200, -83.7913666, 87.6264496
3: -54.3366394, 69.0618134, -41.2748108, 52.2073288, -106.5439606, 110.3366241
4: -54.3550720, 75.1997299, -41.5129852, 56.5219460, -110.8770065, 116.7127151

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8560752, upper bound: 155.8601659
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562720, upper bound: 155.8607391
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -62.2001839, 71.0469208, -51.7201996, 58.8002930, -121.0004730, 122.7671204
1: -47.6223106, 57.1790695, -39.5281334, 47.3404236, -94.9627380, 96.7071991
2: -41.6193466, 55.7893486, -34.5347977, 46.0836258, -87.7029724, 90.3241425
3: -54.3366394, 69.0618134, -45.0400734, 57.0654411, -111.4020844, 114.1018829
4: -54.3550720, 75.1997299, -45.1150551, 61.9955292, -116.3505859, 120.3147888

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8560752, upper bound: 155.8607518
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562720, upper bound: 155.8612770
time: 1.28 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.92 seconds
NS_A1_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
NS_A1_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
NS_A1_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
NS_A1_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
NS_A1_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607319
NS_A1_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607319
NS_A1_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8607031, upper bound: 155.8607630
NS_A1_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8607031, upper bound: 155.8607630
NS_A1_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8607083, upper bound: 155.8628783
NS_A1_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8607083, upper bound: 155.8628783
NS_A1_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8596553, upper bound: 155.8627454
NS_A1_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8596553, upper bound: 155.8627454
NS_A1_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8605191
NS_A1_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8614610
NS_A1_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8605191
NS_A1_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8586350, upper bound: 155.8614661
NS_A1_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8624956, upper bound: 155.8620877
NS_A1_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
NS_A1_B1_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8624956, upper bound: 155.8620877
NS_A1_B1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
NS_A1_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8624945, upper bound: 155.8620877
NS_A1_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
NS_A1_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8624945, upper bound: 155.8620877
NS_A1_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8626100, upper bound: 155.8627747
NS_A1_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8559851, upper bound: 155.8559265
NS_A1_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8555875, upper bound: 155.8559278
NS_A1_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8559851, upper bound: 155.8560566
NS_A1_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8560763, upper bound: 155.8560580
NS_A1_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8560752, upper bound: 155.8601659
NS_A1_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8562720, upper bound: 155.8607391
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8560752, upper bound: 155.8607518
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.92
Output dim: 4, lower bound: -155.8562720, upper bound: 155.8612770

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -34.9739075, 39.2580719, -123.9404984, 130.8989410
1: -64.8533783, 77.4368591, -26.7778168, 31.7589359, -96.6123047, 104.2146606
2: -56.6475601, 75.5014267, -23.3282013, 30.8873501, -87.5349121, 98.8296127
3: -73.5786743, 93.6635971, -30.4137516, 38.3277054, -111.9063797, 124.0773468
4: -73.9747162, 101.9075775, -30.5236435, 41.7227249, -115.6974411, 132.4311981

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -34.9739075, 39.2580719, -121.2936020, 127.1672974
1: -62.6521683, 74.3774872, -26.7778168, 31.7589359, -94.4111023, 101.1553040
2: -54.7120705, 72.4336624, -23.3282013, 30.8873501, -85.5994186, 95.7618561
3: -70.9033813, 89.8668900, -30.4137516, 38.3277054, -109.2310867, 120.2806396
4: -71.3422470, 97.6174927, -30.5236435, 41.7227249, -113.0649719, 128.1411285

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621617
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -35.1993599, 39.5222130, -124.2046509, 131.1243896
1: -64.8533783, 77.4368591, -26.9524441, 31.9716854, -96.8250427, 104.3893051
2: -56.6475601, 75.5014267, -23.4787922, 31.0990868, -87.7466431, 98.9802170
3: -73.5786743, 93.6635971, -30.6249485, 38.5937233, -112.1723938, 124.2885437
4: -73.9747162, 101.9075775, -30.7226028, 42.0155640, -115.9902802, 132.6301575

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8596387, upper bound: 155.8620346
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -35.1993599, 39.5222130, -121.5577545, 127.3927460
1: -62.6521683, 74.3774872, -26.9524441, 31.9716854, -94.6238403, 101.3299332
2: -54.7120705, 72.4336624, -23.4787922, 31.0990868, -85.8111572, 95.9124527
3: -70.9033813, 89.8668900, -30.6249485, 38.5937233, -109.4971008, 120.4918365
4: -71.3422470, 97.6174927, -30.7226028, 42.0155640, -113.3578110, 128.3400879

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -63.1088066, 71.9655151, -36.2489128, 40.7121582, -103.8209686, 108.2144241
1: -48.3480301, 57.9303665, -27.7490921, 32.9438286, -81.2918472, 85.6794510
2: -42.2655525, 56.5074730, -24.1843834, 32.0408592, -74.3064117, 80.6918564
3: -55.1085358, 69.9925842, -31.5226898, 39.7600212, -94.8685455, 101.5152740
4: -55.1945381, 76.1900711, -31.6403008, 43.2684479, -98.4629669, 107.8303680

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607241
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -63.3401642, 72.2599640, -36.2489128, 40.7121582, -104.0523224, 108.5088806
1: -48.5351410, 58.1705399, -27.7490921, 32.9438286, -81.4789505, 85.9196320
2: -42.4297943, 56.7414627, -24.1843834, 32.0408592, -74.4706421, 80.9258423
3: -55.3325043, 70.2826843, -31.5226898, 39.7600212, -95.0925140, 101.8053741
4: -55.4108849, 76.5077286, -31.6403008, 43.2684479, -98.6793365, 108.1480255

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607241
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -63.1088066, 71.9655151, -36.5033760, 41.0126038, -104.1214142, 108.4688644
1: -48.3480301, 57.9303665, -27.9466991, 33.1850510, -81.5330734, 85.8770676
2: -42.2655525, 56.5074730, -24.3541794, 32.2812424, -74.5467987, 80.8616486
3: -55.1085358, 69.9925842, -31.7612324, 40.0626411, -95.1711731, 101.7538147
4: -55.1945381, 76.1900711, -31.8650322, 43.6010666, -98.7955933, 108.0550995

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607552
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -63.3401642, 72.2599640, -36.5033760, 41.0126038, -104.3527679, 108.7633209
1: -48.5351410, 58.1705399, -27.9466991, 33.1850510, -81.7201843, 86.1172333
2: -42.4297943, 56.7414627, -24.3541794, 32.2812424, -74.7110367, 81.0956421
3: -55.3325043, 70.2826843, -31.7612324, 40.0626411, -95.3951416, 102.0439148
4: -55.4108849, 76.5077286, -31.8650322, 43.6010666, -99.0119476, 108.3727570

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604368, upper bound: 155.8607241
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -35.3288994, 39.6333694, -124.3158035, 131.2539368
1: -64.8533783, 77.4368591, -27.0226707, 32.0291748, -96.8825226, 104.4595184
2: -56.6475601, 75.5014267, -23.5637703, 31.1117859, -87.7593460, 99.0651932
3: -73.5786743, 93.6635971, -30.6463623, 38.5703201, -112.1489944, 124.3099518
4: -73.9747162, 101.9075775, -30.7967663, 41.9001198, -115.8748322, 132.7043457

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621550
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621549
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -35.3288994, 39.6333694, -121.6689072, 127.5222931
1: -62.6521683, 74.3774872, -27.0226707, 32.0291748, -94.6813278, 101.4001541
2: -54.7120705, 72.4336624, -23.5637703, 31.1117859, -85.8238525, 95.9974365
3: -70.9033813, 89.8668900, -30.6463623, 38.5703201, -109.4737015, 120.5132446
4: -71.3422470, 97.6174927, -30.7967663, 41.9001198, -113.2423706, 128.4142609

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621550
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606917, upper bound: 155.8621549
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -35.9117203, 40.2799911, -124.9624252, 131.8367615
1: -64.8533783, 77.4368591, -27.4555321, 32.5475578, -97.4009018, 104.8923645
2: -56.6475601, 75.5014267, -23.9453163, 31.6098404, -88.2574005, 99.4467392
3: -73.5786743, 93.6635971, -31.1463699, 39.1984406, -112.7771149, 124.8099670
4: -73.9747162, 101.9075775, -31.2909832, 42.5710640, -116.5457764, 133.1985321

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -35.9117203, 40.2799911, -122.3155289, 128.1051025
1: -62.6521683, 74.3774872, -27.4555321, 32.5475578, -95.1996994, 101.8330002
2: -54.7120705, 72.4336624, -23.9453163, 31.6098404, -86.3219070, 96.3789825
3: -70.9033813, 89.8668900, -31.1463699, 39.1984406, -110.1018219, 121.0132599
4: -71.3422470, 97.6174927, -31.2909832, 42.5710640, -113.9133148, 128.9084473

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -61.4255753, 69.8197250, -53.5605583, 59.2485504, -120.6741257, 123.3802795
1: -47.1093979, 56.2623329, -40.7351456, 48.0194244, -95.1288223, 96.9974747
2: -41.1801262, 54.8647461, -35.4819527, 46.5983582, -87.7784882, 90.3466949
3: -53.6761475, 68.0407333, -45.8934479, 57.9912910, -111.6674347, 113.9341812
4: -53.8180847, 74.0296402, -46.2466049, 62.8137169, -116.6317978, 120.2762451

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -62.0741043, 70.7589798, -53.5605583, 59.2485504, -121.3226547, 124.3195343
1: -47.5669289, 56.9638367, -40.7351456, 48.0194244, -95.5863495, 97.6989517
2: -41.5818863, 55.5577431, -35.4819527, 46.5983582, -88.1802368, 91.0396805
3: -54.2167206, 68.8253403, -45.8934479, 57.9912910, -112.2080078, 114.7187881
4: -54.3048668, 74.9184647, -46.2466049, 62.8137169, -117.1185532, 121.1650696

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -61.4255753, 69.8197250, -34.4380074, 38.9053993, -100.3309784, 104.2577362
1: -47.1093979, 56.2623329, -26.4724121, 31.4414673, -78.5508652, 82.7347412
2: -41.1801262, 54.8647461, -23.0566540, 30.5704288, -71.7505569, 77.9213943
3: -53.6761475, 68.0407333, -30.0729237, 37.8880806, -91.5642242, 98.1136475
4: -53.8180847, 74.0296402, -30.1662598, 41.2300339, -95.0481186, 104.1958923

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -62.0741043, 70.7589798, -34.4380074, 38.9053993, -100.9795074, 105.1969910
1: -47.5669289, 56.9638367, -26.4724121, 31.4414673, -79.0083923, 83.4362335
2: -41.5818863, 55.5577431, -23.0566540, 30.5704288, -72.1523056, 78.6143723
3: -54.2167206, 68.8253403, -30.0729237, 37.8880806, -92.1047974, 98.8982620
4: -54.3048668, 74.9184647, -30.1662598, 41.2300339, -95.5348663, 105.0847092

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -61.8638649, 70.4181137, -63.9723282, 72.7513428, -134.6152039, 134.3904419
1: -47.3781281, 56.7320557, -48.9926109, 58.6176910, -105.9958038, 105.7246704
2: -41.4153328, 55.3562737, -42.8426323, 57.1732521, -98.5885773, 98.1989059
3: -54.0040016, 68.5983582, -55.8395424, 70.8680420, -124.8720398, 124.4378967
4: -54.1098366, 74.6049194, -55.9672127, 77.1005936, -131.2104034, 130.5721283

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8618983
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8621096
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -63.6151657, 72.3106766, -63.9723282, 72.7513428, -136.3665161, 136.2830048
1: -48.7131386, 58.2579384, -48.9926109, 58.6176910, -107.3308258, 107.2505493
2: -42.5970306, 56.8195915, -42.8426323, 57.1732521, -99.7702789, 99.6622238
3: -55.5102272, 70.4305420, -55.8395424, 70.8680420, -126.3782654, 126.2700806
4: -55.6422424, 76.6230087, -55.9672127, 77.1005936, -132.7428284, 132.5902100

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8625854
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8627966
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -61.8638649, 70.4181137, -64.4276733, 73.4758987, -135.3397675, 134.8457947
1: -47.3781281, 56.7320557, -49.2981796, 59.1440315, -106.5221405, 106.0302353
2: -41.4153328, 55.3562737, -43.0962944, 57.6944580, -99.1097870, 98.4525452
3: -54.0040016, 68.5983582, -56.1992683, 71.4468384, -125.4508362, 124.7976227
4: -54.1098366, 74.6049194, -56.2723427, 77.7576904, -131.8675232, 130.8772430

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8618337
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8620877
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -63.6151657, 72.3106766, -64.4276733, 73.4758987, -137.0910645, 136.7383423
1: -48.7131386, 58.2579384, -49.2981796, 59.1440315, -107.8571701, 107.5561218
2: -42.5970306, 56.8195915, -43.0962944, 57.6944580, -100.2914886, 99.9158630
3: -55.5102272, 70.4305420, -56.1992683, 71.4468384, -126.9570618, 126.6298065
4: -55.6422424, 76.6230087, -56.2723427, 77.7576904, -133.3999329, 132.8953247

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8625207
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8627747
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -62.9143600, 71.7894516, -63.9723282, 72.7513428, -135.6657104, 135.7617798
1: -48.1297455, 57.7836342, -48.9926109, 58.6176910, -106.7474213, 106.7762451
2: -42.0624428, 56.3710213, -42.8426323, 57.1732521, -99.2356873, 99.2136459
3: -54.8597221, 69.7908325, -55.8395424, 70.8680420, -125.7277603, 125.6303711
4: -54.9079704, 75.9385300, -55.9672127, 77.1005936, -132.0085602, 131.9057465

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8618983
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8621096
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -64.0428085, 72.9908295, -63.9723282, 72.7513428, -136.7941589, 136.9631653
1: -48.9989014, 58.7516823, -48.9926109, 58.6176910, -107.6165924, 107.7442932
2: -42.8324356, 57.3123703, -42.8426323, 57.1732521, -100.0056763, 100.1549988
3: -55.8455467, 70.9710770, -55.8395424, 70.8680420, -126.7135925, 126.8106232
4: -55.9235916, 77.2398148, -55.9672127, 77.1005936, -133.0241852, 133.2070312

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8621352, upper bound: 155.8621174
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8617923, upper bound: 155.8621171
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -62.9143600, 71.7894516, -64.4276733, 73.4758987, -136.3902588, 136.2171173
1: -48.1297455, 57.7836342, -49.2981796, 59.1440315, -107.2737656, 107.0818176
2: -42.0624428, 56.3710213, -43.0962944, 57.6944580, -99.7568970, 99.4672852
3: -54.8597221, 69.7908325, -56.1992683, 71.4468384, -126.3065567, 125.9900970
4: -54.9079704, 75.9385300, -56.2723427, 77.7576904, -132.6656647, 132.2108765

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8618336
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8620877
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -64.0428085, 72.9908295, -64.4276733, 73.4758987, -137.5187073, 137.4185028
1: -48.9989014, 58.7516823, -49.2981796, 59.1440315, -108.1429291, 108.0498657
2: -42.8324356, 57.3123703, -43.0962944, 57.6944580, -100.5268860, 100.4086533
3: -55.8455467, 70.9710770, -56.1992683, 71.4468384, -127.2923889, 127.1703491
4: -55.9235916, 77.2398148, -56.2723427, 77.7576904, -133.6812744, 133.5121613

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619333, upper bound: 155.8625207
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619333, upper bound: 155.8627747
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -55.3605347, 62.6548157, -47.8892212, 53.9669075, -109.3274384, 110.5440369
1: -42.1221466, 50.3461761, -36.3983994, 43.3684845, -85.4906311, 86.7445755
2: -36.8170700, 49.0921593, -31.8371048, 42.1720200, -78.9890900, 80.9292603
3: -47.8610764, 60.7613525, -41.2748108, 52.2073288, -100.0684052, 102.0361557
4: -48.0124550, 65.9326172, -41.5129852, 56.5219460, -104.5344009, 107.4456024

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8552772, upper bound: 155.8557208
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8546350, upper bound: 155.8559264
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -57.4111252, 64.9420013, -47.8892212, 53.9669075, -111.3780136, 112.8312149
1: -43.6660805, 52.1844788, -36.3983994, 43.3684845, -87.0345612, 88.5828781
2: -38.1825905, 50.8895988, -31.8371048, 42.1720200, -80.3546143, 82.7266998
3: -49.6351128, 62.9960098, -41.2748108, 52.2073288, -101.8424301, 104.2708206
4: -49.7901230, 68.3816910, -41.5129852, 56.5219460, -106.3120499, 109.8946762

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8553681, upper bound: 155.8557223
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8547262, upper bound: 155.8559278
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -55.3605347, 62.6548157, -51.5600166, 58.6148796, -113.9754181, 114.2148285
1: -42.1221466, 50.3461761, -39.4048767, 47.1900711, -89.3122101, 89.7510529
2: -36.8170700, 49.0921593, -34.4266205, 45.9367447, -82.7538147, 83.5187683
3: -47.8610764, 60.7613525, -44.8990669, 56.8835030, -104.7445831, 105.6604156
4: -48.0124550, 65.9326172, -44.9741592, 61.7972603, -109.8097153, 110.9067764

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8574260, upper bound: 155.8557554
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8574260, upper bound: 155.8560566
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -57.4111252, 64.9420013, -51.5600166, 58.6148796, -116.0259857, 116.5020065
1: -43.6660805, 52.1844788, -39.4048767, 47.1900711, -90.8561554, 91.5893555
2: -38.1825905, 50.8895988, -34.4266205, 45.9367447, -84.1193314, 85.3162079
3: -49.6351128, 62.9960098, -44.8990669, 56.8835030, -106.5186157, 107.8950729
4: -49.7901230, 68.3816910, -44.9741592, 61.7972603, -111.5873718, 113.3558502

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586612, upper bound: 155.8560560
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586635, upper bound: 155.8560579
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -60.6136131, 69.2644348, -47.8892212, 53.9669075, -114.5805130, 117.1536407
1: -46.3935738, 55.7460022, -36.3983994, 43.3684845, -89.7620544, 92.1444016
2: -40.5349464, 54.3899651, -31.8371048, 42.1720200, -82.7069626, 86.2270660
3: -52.9255829, 67.3147049, -41.2748108, 52.2073288, -105.1329117, 108.5895157
4: -52.9306259, 73.2776794, -41.5129852, 56.5219460, -109.4525681, 114.7906647

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8553674, upper bound: 155.8601285
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8553674, upper bound: 155.8601659
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -61.8248444, 70.5692062, -47.8892212, 53.9669075, -115.7917480, 118.4584198
1: -47.3268967, 56.7906723, -36.3983994, 43.3684845, -90.6953812, 93.1890717
2: -41.3596992, 55.4098701, -31.8371048, 42.1720200, -83.5317230, 87.2469788
3: -53.9873619, 68.5928955, -41.2748108, 52.2073288, -106.1946793, 109.8676987
4: -54.0109329, 74.6846313, -41.5129852, 56.5219460, -110.5328827, 116.1976166

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8555641, upper bound: 155.8606901
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8555641, upper bound: 155.8607391
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -60.6136131, 69.2644348, -51.7201996, 58.8002930, -119.4139023, 120.9846344
1: -46.3935738, 55.7460022, -39.5281334, 47.3404236, -93.7339783, 95.2741394
2: -40.5349464, 54.3899651, -34.5347977, 46.0836258, -86.6185684, 88.9247589
3: -52.9255829, 67.3147049, -45.0400734, 57.0654411, -109.9910278, 112.3547821
4: -52.9306259, 73.2776794, -45.1150551, 61.9955292, -114.9261475, 118.3927307

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8575127, upper bound: 155.8604716
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8575127, upper bound: 155.8607518
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -61.8248444, 70.5692062, -51.7201996, 58.8002930, -120.6251373, 122.2894058
1: -47.3268967, 56.7906723, -39.5281334, 47.3404236, -94.6673203, 96.3188019
2: -41.3596992, 55.4098701, -34.5347977, 46.0836258, -87.4433289, 89.9446716
3: -53.9873619, 68.5928955, -45.0400734, 57.0654411, -111.0528030, 113.6329651
4: -54.0109329, 74.6846313, -45.1150551, 61.9955292, -116.0064621, 119.7996826

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8577145, upper bound: 155.8610012
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8577145, upper bound: 155.8612770
time: 0.90 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 3.54 seconds
NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8596387, upper bound: 155.8620346
NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8606974, upper bound: 155.8621928
NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621550
NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621549
NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621550
NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8606917, upper bound: 155.8621549
NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8618983
NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8621096
NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8625854
NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8627966
NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8618337
NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618164, upper bound: 155.8620877
NS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8625207
NS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8619342, upper bound: 155.8627747
NS_A1_B1_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8618983
NS_A1_B1_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8621096
NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8621352, upper bound: 155.8621174
NS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8617923, upper bound: 155.8621171
NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8618336
NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8618109, upper bound: 155.8620877
NS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8619333, upper bound: 155.8625207
NS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8619333, upper bound: 155.8627747
NS_A1_B1_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8552772, upper bound: 155.8557208
NS_A1_B1_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8546350, upper bound: 155.8559264
NS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8553681, upper bound: 155.8557223
NS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8547262, upper bound: 155.8559278
NS_A1_B1_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8574260, upper bound: 155.8557554
NS_A1_B1_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8574260, upper bound: 155.8560566
NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8586612, upper bound: 155.8560560
NS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8586635, upper bound: 155.8560579
NS_A1_B1_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8553674, upper bound: 155.8601285
NS_A1_B1_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8553674, upper bound: 155.8601659
NS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8555641, upper bound: 155.8606901
NS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8555641, upper bound: 155.8607391
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8575127, upper bound: 155.8604716
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8575127, upper bound: 155.8607518
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8577145, upper bound: 155.8610012
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.54
Output dim: 4, lower bound: -155.8577145, upper bound: 155.8612770

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -34.2308960, 38.4290237, -123.1114655, 130.1559296
1: -64.8533783, 77.4368591, -26.2174492, 31.0910358, -95.9444046, 103.6543121
2: -56.6475601, 75.5014267, -22.8370781, 30.2445602, -86.8921204, 98.3384857
3: -73.5786743, 93.6635971, -29.7861671, 37.5238495, -111.1025238, 123.4497604
4: -73.9747162, 101.9075775, -29.8896198, 40.8744545, -114.8491669, 131.7971954

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -34.8122330, 39.0612106, -123.7436523, 130.7372742
1: -64.8533783, 77.4368591, -26.6551743, 31.5976849, -96.4510498, 104.0920181
2: -56.6475601, 75.5014267, -23.2195244, 30.7313480, -87.3789062, 98.7209473
3: -73.5786743, 93.6635971, -30.2791042, 38.1380234, -111.7166901, 123.9427032
4: -73.9747162, 101.9075775, -30.3811283, 41.5197487, -115.4944611, 132.2886963

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -55.0806313, 60.9678574, -143.0033569, 147.2740173
1: -62.6521683, 74.3774872, -42.0122910, 49.5000381, -112.1522064, 116.3897705
2: -54.7120705, 72.4336624, -36.5560608, 48.0731087, -102.7851791, 108.9897232
3: -70.9033813, 89.8668900, -47.2731247, 60.0138245, -130.9172058, 137.1399994
4: -71.3422470, 97.6174927, -47.6950264, 65.0275040, -136.3697357, 145.3125153

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -32.8155556, 37.1126480, -119.1481857, 125.0089417
1: -62.6521683, 74.3774872, -25.2803650, 30.0381317, -92.6902847, 99.6578522
2: -54.7120705, 72.4336624, -21.9997864, 29.2631645, -83.9752350, 94.4334488
3: -70.9033813, 89.8668900, -28.7992611, 36.2907066, -107.1940918, 118.6661530
4: -71.3422470, 97.6174927, -28.8205605, 39.6331787, -110.9754257, 126.4380493

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -35.1041832, 39.3837509, -124.0661774, 131.0292053
1: -64.8533783, 77.4368591, -26.8491287, 31.8257580, -96.6791229, 104.2859879
2: -56.6475601, 75.5014267, -23.4117546, 30.9157734, -87.5633316, 98.9131699
3: -73.5786743, 93.6635971, -30.4506111, 38.3238373, -111.9025116, 124.1142044
4: -73.9747162, 101.9075775, -30.5986481, 41.6388359, -115.6135559, 132.5062103

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604567, upper bound: 155.8621228
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604567, upper bound: 155.8621550
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -84.6824417, 95.9250336, -35.2703705, 39.5662270, -124.2486572, 131.1954041
1: -64.8533783, 77.4368591, -26.9727345, 31.9732075, -96.8265839, 104.4095917
2: -56.6475601, 75.5014267, -23.5180054, 31.0586319, -87.7061920, 99.0194321
3: -73.5786743, 93.6635971, -30.5942497, 38.5086441, -112.0873184, 124.2578430
4: -73.9747162, 101.9075775, -30.7400284, 41.8348656, -115.8095856, 132.6475983

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607174, upper bound: 155.8621226
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607174, upper bound: 155.8621549
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -35.1041832, 39.3837509, -121.4192810, 127.2975769
1: -62.6521683, 74.3774872, -26.8491287, 31.8257580, -94.4779129, 101.2266159
2: -54.7120705, 72.4336624, -23.4117546, 30.9157734, -85.6278458, 95.8454132
3: -70.9033813, 89.8668900, -30.4506111, 38.3238373, -109.2272186, 120.3175049
4: -71.3422470, 97.6174927, -30.5986481, 41.6388359, -112.9810791, 128.2161407

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621549
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604310, upper bound: 155.8621549
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -82.0355377, 92.1933899, -35.2703705, 39.5662270, -121.6017609, 127.4637604
1: -62.6521683, 74.3774872, -26.9727345, 31.9732075, -94.6253738, 101.3502197
2: -54.7120705, 72.4336624, -23.5180054, 31.0586319, -85.7706985, 95.9516678
3: -70.9033813, 89.8668900, -30.5942497, 38.5086441, -109.4120255, 120.4611359
4: -71.3422470, 97.6174927, -30.7400284, 41.8348656, -113.1771088, 128.3574982

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606917, upper bound: 155.8621549
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606917, upper bound: 155.8621549
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -61.8638649, 70.4181137, -61.8638649, 70.4181137, -132.2819672, 132.2819519
1: -47.3781281, 56.7320557, -47.3781281, 56.7320557, -104.1101685, 104.1101761
2: -41.4153328, 55.3562737, -41.4153328, 55.3562737, -96.7715912, 96.7716064
3: -54.0040016, 68.5983582, -54.0040016, 68.5983582, -122.6023560, 122.6023560
4: -54.1098366, 74.6049194, -54.1098366, 74.6049194, -128.7147522, 128.7147522

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8617281, upper bound: 155.8618521
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8616762, upper bound: 155.8617916
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -61.8638649, 70.4181137, -63.6151657, 72.3106766, -134.1745148, 134.0332794
1: -47.3781281, 56.7320557, -48.7131386, 58.2579384, -105.6360626, 105.4451904
2: -41.4153328, 55.3562737, -42.5970306, 56.8195915, -98.2349167, 97.9533005
3: -54.0040016, 68.5983582, -55.5102272, 70.4305420, -124.4345398, 124.1085815
4: -54.1098366, 74.6049194, -55.6422424, 76.6230087, -130.7328186, 130.2471619

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8617281, upper bound: 155.8620630
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8616762, upper bound: 155.8620026
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -63.6151657, 72.3106766, -61.8638649, 70.4181137, -134.0332794, 134.1745300
1: -48.7131386, 58.2579384, -47.3781281, 56.7320557, -105.4451904, 105.6360626
2: -42.5970306, 56.8195915, -41.4153328, 55.3562737, -97.9533005, 98.2349243
3: -55.5102272, 70.4305420, -54.0040016, 68.5983582, -124.1085815, 124.4345398
4: -55.6422424, 76.6230087, -54.1098366, 74.6049194, -130.2471619, 130.7328339

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618517, upper bound: 155.8625538
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8616208, upper bound: 155.8619062
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -63.6151657, 72.3106766, -63.6151657, 72.3106766, -135.9258423, 135.9258423
1: -48.7131386, 58.2579384, -48.7131386, 58.2579384, -106.9710770, 106.9710770
2: -42.5970306, 56.8195915, -42.5970306, 56.8195915, -99.4166183, 99.4166183
3: -55.5102272, 70.4305420, -55.5102272, 70.4305420, -125.9407654, 125.9407654
4: -55.6422424, 76.6230087, -55.6422424, 76.6230087, -132.2652435, 132.2652435

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618517, upper bound: 155.8627131
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8616208, upper bound: 155.8620159
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -61.8638649, 70.4181137, -62.9143600, 71.7894516, -133.6532745, 133.3324585
1: -47.3781281, 56.7320557, -48.1297455, 57.7836342, -105.1617508, 104.8617935
2: -41.4153328, 55.3562737, -42.0624428, 56.3710213, -97.7863312, 97.4187012
3: -54.0040016, 68.5983582, -54.8597221, 69.7908325, -123.7948303, 123.4580765
4: -54.1098366, 74.6049194, -54.9079704, 75.9385300, -130.0483704, 129.5128784

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8617281, upper bound: 155.8617862
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -61.8638649, 70.4181137, -64.0428085, 72.9908295, -134.8546753, 134.4609222
1: -47.3781281, 56.7320557, -48.9989014, 58.7516823, -106.1298065, 105.7309570
2: -41.4153328, 55.3562737, -42.8324356, 57.3123703, -98.7277069, 98.1886902
3: -54.0040016, 68.5983582, -55.8455467, 70.9710770, -124.9750824, 124.4439087
4: -54.1098366, 74.6049194, -55.9235916, 77.2398148, -131.3496399, 130.5285034

Time for backsubstitution: 1.44 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.63 + 416.86 = 420.50 seconds
