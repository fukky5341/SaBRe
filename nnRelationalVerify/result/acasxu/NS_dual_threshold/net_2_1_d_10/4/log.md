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
execution time: IAR + RelationalAnalysis = 1.25 + 2.29 = 3.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -155.8676494, upper bound: 155.8676494

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8630827, upper bound: 155.8584031
time: 1.36 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 1.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.76 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.76
Output dim: 4, lower bound: -155.8630827, upper bound: 155.8584031
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.76
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -81.3169098, 96.3277130, -60.2190781, 69.0278397, -150.3447418, 156.5467834
1: -62.8572426, 77.7227859, -46.0649567, 55.6417961, -118.4990234, 123.7877350
2: -55.0245743, 76.3968430, -40.2784386, 54.2545853, -109.2791595, 116.6752777
3: -72.9154510, 93.5727158, -52.6274223, 67.0956802, -140.0111389, 146.2001190
4: -72.2197571, 103.2277756, -52.6573296, 73.0029373, -145.2226868, 155.8850861

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 0.92 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 0.89 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -75.6169586, 89.3531570, -108.3899231, 133.7219238, -209.3388824, 197.7430267
1: -58.4705429, 72.1841736, -85.1367645, 107.0074463, -165.4779968, 157.3209381
2: -51.1664467, 70.8636856, -74.3936386, 106.0023499, -157.1687927, 145.2573242
3: -67.6475906, 87.0211868, -99.2187576, 129.2454376, -196.8930359, 186.2399445
4: -67.1893158, 95.6991959, -98.1038055, 142.7249908, -209.9143066, 193.8029938

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 1.04 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
time: 1.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.46 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -155.8569978, upper bound: 155.8569978

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -60.2190781, 69.0278397, -60.2190781, 69.0278397, -129.2469025, 129.2469025
1: -46.0649567, 55.6417961, -46.0649567, 55.6417961, -101.7067337, 101.7067337
2: -40.2784386, 54.2545853, -40.2784386, 54.2545853, -94.5329895, 94.5329895
3: -52.6274223, 67.0956802, -52.6274223, 67.0956802, -119.7230988, 119.7230988
4: -52.6573296, 73.0029373, -52.6573296, 73.0029373, -125.6602631, 125.6602631

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8500387, upper bound: 155.8399396
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8500387, upper bound: 155.8581304
time: 1.03 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -108.3899231, 133.7219238, -60.2190781, 69.0278397, -177.4177246, 193.9409943
1: -85.1367645, 107.0074463, -46.0649567, 55.6417961, -140.7785492, 153.0724030
2: -74.3936386, 106.0023499, -40.2784386, 54.2545853, -128.6481934, 146.2807922
3: -99.2187576, 129.2454376, -52.6274223, 67.0956802, -166.3144379, 181.8728638
4: -98.1038055, 142.7249908, -52.6573296, 73.0029373, -171.1067505, 195.3823090

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8526818
time: 0.69 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8526818
time: 1.09 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -60.2190781, 69.0278397, -108.3899231, 133.7219238, -193.9409943, 177.4177246
1: -46.0649567, 55.6417961, -85.1367645, 107.0074463, -153.0724030, 140.7785645
2: -40.2784386, 54.2545853, -74.3936386, 106.0023499, -146.2807922, 128.6481781
3: -52.6274223, 67.0956802, -99.2187576, 129.2454376, -181.8728638, 166.3144379
4: -52.6573296, 73.0029373, -98.1038055, 142.7249908, -195.3823090, 171.1067505

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8389050
time: 1.03 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567250, upper bound: 155.8567250
time: 1.09 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -108.3899231, 133.7219238, -108.3899231, 133.7219238, -242.1118164, 242.1118164
1: -85.1367645, 107.0074463, -85.1367645, 107.0074463, -192.1442108, 192.1442108
2: -74.3936386, 106.0023499, -74.3936386, 106.0023499, -180.3959961, 180.3959961
3: -99.2187576, 129.2454376, -99.2187576, 129.2454376, -228.4642029, 228.4642029
4: -98.1038055, 142.7249908, -98.1038055, 142.7249908, -240.8287964, 240.8287964

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8477394, upper bound: 155.8553327
time: 0.84 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8473447, upper bound: 155.8474432
time: 0.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.28 seconds
NS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8500387, upper bound: 155.8399396
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8500387, upper bound: 155.8581304
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8526818
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8528078, upper bound: 155.8526818
NS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8469703, upper bound: 155.8389050
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8567250, upper bound: 155.8567250
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8477394, upper bound: 155.8553327
NS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 4, lower bound: -155.8473447, upper bound: 155.8474432

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -59.8527756, 68.5774918, -126.5137482, 126.0706177
1: -44.3237762, 53.3550873, -45.7839432, 55.2742844, -99.5980606, 99.1390305
2: -38.7481384, 51.9732895, -40.0318985, 53.8888206, -92.6369553, 92.0051880
3: -50.5645981, 64.3627167, -52.2944145, 66.6559219, -117.2205200, 116.6571274
4: -50.6411171, 69.9437561, -52.3323059, 72.5115967, -123.1527100, 122.2760620

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634050
time: 0.98 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634296
time: 0.97 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -105.7787018, 130.1714935, -40.9217491, 46.0795670, -151.8582153, 171.0932465
1: -83.0240326, 104.1414490, -31.2777653, 37.2222519, -120.2462845, 135.4192200
2: -72.5307236, 103.1036530, -27.3006096, 36.1713181, -108.7020416, 130.4042511
3: -96.6085434, 125.8380051, -35.5017929, 44.8549156, -141.4634552, 161.3397675
4: -95.6322021, 138.7948914, -35.6733818, 48.6733131, -144.3055115, 174.4682465

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8444229, upper bound: 155.8500066
time: 1.06 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8435508, upper bound: 155.8465010
time: 0.94 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -108.0825806, 133.3024902, -57.9362602, 66.2178497, -174.3004303, 191.2387238
1: -84.8943481, 106.6633301, -44.3237762, 53.3550873, -138.2494354, 150.9871063
2: -74.1782150, 105.6524048, -38.7481384, 51.9732895, -126.1515045, 144.4005280
3: -98.9163666, 128.8399963, -50.5645981, 64.3627167, -163.2790527, 179.4045563
4: -97.8169022, 142.2529449, -50.6411171, 69.9437561, -167.7606506, 192.8940582

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8617025, upper bound: 155.8567428
time: 1.22 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8617025, upper bound: 155.8581304
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -108.0825806, 133.3024902, -191.2387238, 174.3004303
1: -44.3237762, 53.3550873, -84.8943481, 106.6633301, -150.9871063, 138.2494354
2: -38.7481384, 51.9732895, -74.1782150, 105.6524048, -144.4005280, 126.1515045
3: -50.5645981, 64.3627167, -98.9163666, 128.8399963, -179.4045563, 163.2790527
4: -50.6411171, 69.9437561, -97.8169022, 142.2529449, -192.8940582, 167.7606506

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567428, upper bound: 155.8617025
time: 1.13 seconds

## Relational analysis of NS_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8500387
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8617025
time: 1.18 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -114.7469864, 140.1150665, -105.2925949, 129.7992249, -244.5462036, 245.4076538
1: -89.6251984, 111.8028870, -82.7332687, 103.8465500, -193.4717407, 194.5361633
2: -78.3454819, 110.8382721, -72.2754822, 102.8404465, -181.1859283, 183.1137543
3: -104.3714523, 134.9422607, -96.3308334, 125.4817734, -229.8532257, 231.2730865
4: -103.0620575, 149.4469147, -95.3177414, 138.4638672, -241.5259247, 244.7646332

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8457625, upper bound: 155.8463782
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8457531, upper bound: 155.8463782
time: 1.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.36 seconds
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634050
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8613431, upper bound: 155.8634296
NS_B1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8444229, upper bound: 155.8500066
NS_B1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8435508, upper bound: 155.8465010
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8617025, upper bound: 155.8567428
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8617025, upper bound: 155.8581304
NS_B2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8500387
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8617025
NS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8457625, upper bound: 155.8463782
NS_B2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8457531, upper bound: 155.8463782

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -40.9217491, 46.0795670, -104.0158234, 107.1395874
1: -44.3237762, 53.3550873, -31.2777653, 37.2222519, -81.5460281, 84.6328430
2: -38.7481384, 51.9732895, -27.3006096, 36.1713181, -74.9194489, 79.2738953
3: -50.5645981, 64.3627167, -35.5017929, 44.8549156, -95.4195023, 99.8645096
4: -50.6411171, 69.9437561, -35.6733818, 48.6733131, -99.3144226, 105.6171417

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613279, upper bound: 155.8634050
time: 1.15 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613279, upper bound: 155.8633955
time: 1.03 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -57.9362602, 66.2178497, -124.1541061, 124.1541138
1: -44.3237762, 53.3550873, -44.3237762, 53.3550873, -97.6788635, 97.6788635
2: -38.7481384, 51.9732895, -38.7481384, 51.9732895, -90.7214279, 90.7214279
3: -50.5645981, 64.3627167, -50.5645981, 64.3627167, -114.9273071, 114.9273071
4: -50.6411171, 69.9437561, -50.6411171, 69.9437561, -120.5848694, 120.5848694

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8595490, upper bound: 155.8586748
time: 1.11 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8595490, upper bound: 155.8622869
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -108.0825806, 133.3024902, -55.5051880, 63.2402725, -171.3228455, 188.8076782
1: -84.8943481, 106.6633301, -42.4021111, 51.0330620, -135.9273987, 149.0654449
2: -74.1782150, 105.6524048, -37.0842018, 49.7446289, -123.9228439, 142.7365570
3: -98.9163666, 128.8399963, -48.4690247, 61.5871696, -160.5034790, 177.3089752
4: -97.8169022, 142.2529449, -48.4467201, 67.0056458, -164.8225403, 190.6996613

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B1_B1

### Relational analysis result of NS_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8606548, upper bound: 155.8565301
time: 1.14 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2

### Relational analysis result of NS_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8615328, upper bound: 155.8565655
time: 1.15 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -108.0825806, 133.3024902, -56.4791565, 64.5003052, -172.5828857, 189.7816162
1: -84.8943481, 106.6633301, -43.2086639, 51.9795380, -136.8738708, 149.8719940
2: -74.1782150, 105.6524048, -37.7727242, 50.6243019, -124.8025208, 143.4251251
3: -98.9163666, 128.8399963, -49.2845078, 62.6991501, -161.6154938, 178.1244965
4: -97.8169022, 142.2529449, -49.3638763, 68.1325607, -165.9494629, 191.6168213

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8455876, upper bound: 155.8510080
time: 1.08 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8445439, upper bound: 155.8465062
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -106.1845322, 130.7253113, -188.6615448, 172.4023285
1: -44.3237762, 53.3550873, -83.3945312, 104.5468597, -148.8706360, 136.7496185
2: -38.7481384, 51.9732895, -72.8464279, 103.5035629, -142.2517090, 124.8197174
3: -50.5645981, 64.3627167, -97.0476532, 126.3464890, -176.9110565, 161.4103546
4: -50.6411171, 69.9437561, -96.0440063, 139.3540802, -189.9951935, 165.9877625

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8556981
time: 1.17 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8383486, upper bound: 155.8445434
time: 1.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.41 seconds
NS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8613279, upper bound: 155.8634050
NS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8613279, upper bound: 155.8633955
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8595490, upper bound: 155.8586748
NS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8595490, upper bound: 155.8622869
NS_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8606548, upper bound: 155.8565301
NS_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8615328, upper bound: 155.8565655
NS_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8455876, upper bound: 155.8510080
NS_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8445439, upper bound: 155.8465062
NS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8399396, upper bound: 155.8556981
NS_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.41
Output dim: 4, lower bound: -155.8383486, upper bound: 155.8445434

## BFS NS instance: NS_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -38.6781540, 43.5407829, -101.4770432, 104.8960037
1: -44.3237762, 53.3550873, -29.5861378, 35.2366562, -79.5604324, 82.9412231
2: -38.7481384, 51.9732895, -25.8011398, 34.2836494, -73.0317841, 77.7744293
3: -50.5645981, 64.3627167, -33.6428108, 42.5293121, -93.0938950, 98.0055237
4: -50.6411171, 69.9437561, -33.7602158, 46.2752151, -96.9163361, 103.7039719

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B1_B1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8587667, upper bound: 155.8620229
time: 1.00 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597101, upper bound: 155.8622544
time: 1.26 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -39.8205414, 44.8232765, -102.7595367, 106.0383911
1: -44.3237762, 53.3550873, -30.4353352, 36.2109261, -80.5346985, 83.7904205
2: -38.7481384, 51.9732895, -26.5637455, 35.1901360, -73.9382782, 78.5370331
3: -50.5645981, 64.3627167, -34.5487022, 43.6286316, -94.1932068, 98.9114075
4: -50.6411171, 69.9437561, -34.7117538, 47.3607826, -98.0018997, 104.6555099

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8612931, upper bound: 155.8633955
time: 1.08 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613279, upper bound: 155.8630340
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -50.8869286, 57.6407890, -54.1052017, 61.8495255, -112.7364502, 111.7459869
1: -38.7363510, 46.3467331, -41.4054985, 49.8134308, -88.5497818, 87.7522278
2: -33.8911552, 45.0993500, -36.1905708, 48.5275726, -82.4187317, 81.2899170
3: -44.0062294, 55.8266296, -47.2375259, 60.0788345, -104.0850525, 103.0641327
4: -44.2170410, 60.5018501, -47.3045006, 65.3020630, -109.5190887, 107.8063278

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8611446, upper bound: 155.8586595
time: 0.86 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8609809, upper bound: 155.8578869
time: 1.09 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8591961, upper bound: 155.8585244
time: 1.25 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -54.9811630, 62.7942581, -57.9362602, 66.2178497, -121.1989975, 120.7305145
1: -42.0638466, 50.6070824, -44.3237762, 53.3550873, -95.4189301, 94.9308548
2: -36.7710495, 49.2830200, -38.7481384, 51.9732895, -88.7443390, 88.0311584
3: -48.0106544, 61.0226974, -50.5645981, 64.3627167, -112.3733673, 111.5872879
4: -48.0589600, 66.3295975, -50.6411171, 69.9437561, -118.0027161, 116.9707184

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8618351, upper bound: 155.8617339
time: 1.08 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8619481, upper bound: 155.8621755
time: 1.00 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -108.0825806, 133.3024902, -54.9507790, 62.5415230, -170.6240997, 188.2532501
1: -84.8943481, 106.6633301, -41.9488029, 50.4661179, -135.3604736, 148.6121368
2: -74.1782150, 105.6524048, -36.6837845, 49.1830635, -123.3612823, 142.3361511
3: -98.9163666, 128.8399963, -47.9240990, 60.8931389, -159.8094788, 176.7640839
4: -97.8169022, 142.2529449, -47.9199638, 66.2131500, -164.0300293, 190.1729126

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_B1_B1_B1

### Relational analysis result of NS_B1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597854, upper bound: 155.8534715
time: 1.11 seconds

## Relational analysis of NS_B1_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604656, upper bound: 155.8556391
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8605470, upper bound: 155.8558137
time: 0.97 seconds

## Relational analysis of NS_B1_A2_B2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8605470, upper bound: 155.8565299
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -108.0825806, 133.3024902, -54.9260063, 62.5169563, -170.5995331, 188.2285004
1: -84.8943481, 106.6633301, -41.9475060, 50.4442406, -135.3385620, 148.6108398
2: -74.1782150, 105.6524048, -36.6861153, 49.1667213, -123.3449249, 142.3385162
3: -98.9163666, 128.8399963, -47.9329758, 60.8721771, -159.7885132, 176.7729797
4: -97.8169022, 142.2529449, -47.9165764, 66.2244720, -164.0413666, 190.1695251

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8604656, upper bound: 155.8539344
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613462, upper bound: 155.8558377
time: 1.12 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8613462, upper bound: 155.8565655
time: 1.31 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -51.6664810, 58.6859055, -106.1845322, 130.7253113, -182.3917847, 164.8704224
1: -39.4619751, 47.2641678, -83.3945312, 104.5468597, -144.0088348, 130.6586914
2: -34.4859314, 45.9825592, -72.8464279, 103.5035629, -137.9895020, 118.8289871
3: -44.9092712, 56.9975624, -97.0476532, 126.3464890, -171.2557678, 154.0452118
4: -45.0333710, 61.8690224, -96.0440063, 139.3540802, -184.3874512, 157.9129944

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434
time: 1.07 seconds

## Relational analysis of NS_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434
time: 3.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.49 seconds
NS_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8587667, upper bound: 155.8620229
NS_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8597101, upper bound: 155.8622544
NS_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8612931, upper bound: 155.8633955
NS_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8613279, upper bound: 155.8630340
NS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8609809, upper bound: 155.8578869
NS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8591961, upper bound: 155.8585244
NS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8618351, upper bound: 155.8617339
NS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8619481, upper bound: 155.8621755
NS_B1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8605470, upper bound: 155.8558137
NS_B1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8605470, upper bound: 155.8565299
NS_B1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8613462, upper bound: 155.8558377
NS_B1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8613462, upper bound: 155.8565655
NS_B2_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434
NS_B2_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.49
Output dim: 4, lower bound: -155.8445471, upper bound: 155.8445434

## BFS NS instance: NS_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -54.1052017, 61.8495255, -33.9881058, 37.8458290, -91.9510345, 95.8376236
1: -41.4054985, 49.8134308, -25.8275013, 30.5502663, -71.9557571, 75.6409302
2: -36.1905708, 48.5275726, -22.5280457, 29.7491150, -65.9396820, 71.0556183
3: -47.2375259, 60.0788345, -29.2773724, 36.7953606, -84.0328598, 89.3562012
4: -47.3045006, 65.3020630, -29.4040413, 39.9082985, -87.2127991, 94.7061005

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8587280, upper bound: 155.8618831
time: 1.01 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8587667, upper bound: 155.8617647
time: 0.96 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -57.9362602, 66.2178497, -36.5424843, 41.1402893, -99.0765457, 102.7603226
1: -44.3237762, 53.3550873, -27.9899483, 33.2968178, -77.6205750, 81.3450317
2: -38.7481384, 51.9732895, -24.4130478, 32.3903999, -71.1385345, 76.3863297
3: -50.5645981, 64.3627167, -31.8436470, 40.1727753, -90.7373657, 96.2063522
4: -50.6411171, 69.9437561, -31.9443474, 43.7624702, -94.4035873, 101.8880997

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8596714, upper bound: 155.8622544
time: 0.99 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8587667, upper bound: 155.8620740
time: 0.99 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -80.1093979, 89.9962540, -38.2738762, 43.0429382, -123.1523361, 128.2701263
1: -61.0311050, 72.7571487, -29.2516880, 34.7682991, -95.7994080, 102.0088348
2: -53.3270264, 70.7531281, -25.5270023, 33.7846107, -87.1116257, 96.2801285
3: -69.1591568, 87.7683868, -33.2049828, 41.8825607, -111.0417175, 120.9733658
4: -69.4885101, 95.2814255, -33.3565445, 45.4812088, -114.9697113, 128.6379700

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8572097, upper bound: 155.8613095
time: 1.11 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8596714, upper bound: 155.8622528
time: 1.17 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -54.5643578, 62.3822823, -39.8205414, 44.8232765, -99.3876343, 102.2028198
1: -41.8142624, 50.2631187, -30.4353352, 36.2109261, -78.0251770, 80.6984558
2: -36.5434074, 48.9434204, -26.5637455, 35.1901360, -71.7335434, 75.5071411
3: -47.6924667, 60.6391754, -34.5487022, 43.6286316, -91.3210907, 95.1878738
4: -47.7632027, 65.9152145, -34.7117538, 47.3607826, -95.1239853, 100.6269684

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8581432, upper bound: 155.8610792
time: 1.22 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597101, upper bound: 155.8620724
time: 1.01 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -50.1599197, 56.5729485, -54.1052017, 61.8495255, -112.0094452, 110.6781387
1: -38.1266136, 45.4917221, -41.4054985, 49.8134308, -87.9400406, 86.8972168
2: -33.3535233, 44.2322769, -36.1905708, 48.5275726, -81.8810883, 80.4228516
3: -43.2467461, 54.7902069, -47.2375259, 60.0788345, -103.3255768, 102.0277328
4: -43.4888687, 59.3089828, -47.3045006, 65.3020630, -108.7909241, 106.6134720

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8585982, upper bound: 155.8560566
time: 0.96 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8585567, upper bound: 155.8556143
time: 1.06 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -50.3368797, 56.9905090, -54.1052017, 61.8495255, -112.1864014, 111.0957108
1: -38.3109703, 45.8174400, -41.4054985, 49.8134308, -88.1244049, 87.2229309
2: -33.5184898, 44.5838814, -36.1905708, 48.5275726, -82.0460510, 80.7744370
3: -43.5130882, 55.1820412, -47.2375259, 60.0788345, -103.5919037, 102.4195480
4: -43.7251892, 59.8008118, -47.3045006, 65.3020630, -109.0272446, 107.1053009

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586635, upper bound: 155.8560580
time: 1.28 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8579688, upper bound: 155.8563185
time: 0.99 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -54.5236549, 62.1283493, -57.9362602, 66.2178497, -120.7414932, 120.0646057
1: -41.6839180, 50.0847168, -44.3237762, 53.3550873, -95.0390015, 94.4084930
2: -36.4332275, 48.7508049, -38.7481384, 51.9732895, -88.4065094, 87.4989471
3: -47.5359039, 60.3813553, -50.5645981, 64.3627167, -111.8986130, 110.9459381
4: -47.6024246, 65.5965042, -50.6411171, 69.9437561, -117.5461731, 116.2376251

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_A2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586884, upper bound: 155.8607518
time: 0.77 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586341, upper bound: 155.8577648
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -54.3511848, 62.0404854, -57.9362602, 66.2178497, -120.5690308, 119.9767380
1: -41.5749817, 49.9943504, -44.3237762, 53.3550873, -94.9300690, 94.3181229
2: -36.3425407, 48.6838341, -38.7481384, 51.9732895, -88.3158188, 87.4319763
3: -47.4417801, 60.2769547, -50.5645981, 64.3627167, -111.8044968, 110.8415527
4: -47.4920387, 65.5164261, -50.6411171, 69.9437561, -117.4357910, 116.1575394

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A2_B2_A2_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8462054, upper bound: 155.8540716
time: 1.38 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8440586, upper bound: 155.8441240
time: 1.07 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -107.5470657, 132.6109772, -54.9507790, 62.5415230, -170.0885925, 187.5617218
1: -84.4782181, 106.0963669, -41.9488029, 50.4661179, -134.9443359, 148.0451660
2: -73.8121643, 105.0893936, -36.6837845, 49.1830635, -122.9952240, 141.7731781
3: -98.4116516, 128.1703186, -47.9240990, 60.8931389, -159.3047791, 176.0944214
4: -97.3327560, 141.4856720, -47.9199638, 66.2131500, -163.5458679, 189.4056396

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

## BFS NS instance: NS_B1_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -107.2810364, 132.3269958, -54.9507790, 62.5415230, -169.8225403, 187.2777557
1: -84.2750092, 105.8871994, -41.9488029, 50.4661179, -134.7411194, 147.8359985
2: -73.6324463, 104.8832550, -36.6837845, 49.1830635, -122.8155060, 141.5670471
3: -98.1859589, 127.9107208, -47.9240990, 60.8931389, -159.0790710, 175.8347931
4: -97.1040344, 141.2091827, -47.9199638, 66.2131500, -163.3171844, 189.1291504

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

## BFS NS instance: NS_B1_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -107.5470657, 132.6109772, -54.9260063, 62.5169563, -170.0640106, 187.5369568
1: -84.4782181, 106.0963669, -41.9475060, 50.4442406, -134.9224243, 148.0438538
2: -73.8121643, 105.0893936, -36.6861153, 49.1667213, -122.9788742, 141.7755127
3: -98.4116516, 128.1703186, -47.9329758, 60.8721771, -159.2837982, 176.1033020
4: -97.3327560, 141.4856720, -47.9165764, 66.2244720, -163.5571899, 189.4022522

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

## BFS NS instance: NS_B1_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -107.2810364, 132.3269958, -54.9260063, 62.5169563, -169.7979736, 187.2529907
1: -84.2750092, 105.8871994, -41.9475060, 50.4442406, -134.7192230, 147.8346710
2: -73.6324463, 104.8832550, -36.6861153, 49.1667213, -122.7991562, 141.5693665
3: -98.1859589, 127.9107208, -47.9329758, 60.8721771, -159.0581207, 175.8436890
4: -97.1040344, 141.2091827, -47.9165764, 66.2244720, -163.3285065, 189.1257477

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.54 + 134.52 = 138.06 seconds
