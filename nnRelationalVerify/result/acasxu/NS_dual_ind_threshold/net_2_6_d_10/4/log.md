## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 56.43210397135999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-150.7717438, 278.2929382, -150.7717438, 278.2929382, -429.0646973, 429.0646973)
1: (-30.6839504, 35.1584358, -30.6839504, 35.1584358, -65.8423691, 65.8423767)
2: (-21.8219566, 36.0254250, -21.8219566, 36.0254250, -57.8473816, 57.8473816)
3: (-20.6292686, 60.2387085, -20.6292686, 60.2387085, -80.8679810, 80.8679810)
4: (-17.8182507, 44.7172356, -17.8182507, 44.7172356, -62.5354805, 62.5354805)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.38 + 1.71 = 4.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -56.4772858, upper bound: 56.4772858

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4695637, upper bound: 56.4353945
time: 0.62 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4740145, upper bound: 56.4740144
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.43 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 4, lower bound: -56.4695637, upper bound: 56.4353945
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 4, lower bound: -56.4740145, upper bound: 56.4740144

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -120.2060547, 217.4064484, -145.4526215, 267.8081970, -388.0142517, 362.8590698
1: -23.8955574, 27.5374279, -29.5412064, 33.8290443, -57.7246017, 57.0786362
2: -17.0973358, 28.4529037, -21.0201759, 34.7042999, -51.8016357, 49.4730759
3: -16.3024616, 47.5721283, -19.9048100, 58.0669403, -74.3693848, 67.4769363
4: -13.9706097, 35.3413849, -17.1677170, 43.0479546, -57.0185623, 52.5091019

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4304713, upper bound: 56.4304713
time: 0.57 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4304713, upper bound: 56.4351857
time: 0.52 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -278.9472961, 522.5424805, -144.9349060, 266.3251038, -545.2723999, 667.4774170
1: -58.4989815, 66.3320999, -29.3481979, 33.6743050, -92.1732788, 95.6802902
2: -41.0573997, 67.8399963, -20.9082088, 34.5005989, -75.5579758, 88.7481995
3: -39.0714874, 112.9678192, -19.7824478, 57.7535172, -96.8250046, 132.7502136
4: -33.8307724, 83.7769852, -17.0621777, 42.8435326, -76.6743011, 100.8391647

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4351857, upper bound: 56.4693103
time: 0.59 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4351857, upper bound: 56.4740145
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.62 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.62
Output dim: 4, lower bound: -56.4304713, upper bound: 56.4304713
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -56.4304713, upper bound: 56.4351857
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -56.4351857, upper bound: 56.4693103
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -56.4351857, upper bound: 56.4740145

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -120.2060547, 217.4064484, -278.9472961, 522.5424805, -642.7485352, 496.3537598
1: -23.8955574, 27.5374279, -58.4989815, 66.3320999, -90.2276611, 86.0364075
2: -17.0973358, 28.4529037, -41.0573997, 67.8399963, -84.9373245, 69.5102997
3: -16.3024616, 47.5721283, -39.0714874, 112.9678192, -129.2702484, 86.6436157
4: -13.9706097, 35.3413849, -33.8307724, 83.7769852, -97.7475967, 69.1721573

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4257477, upper bound: 56.4242232
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4258014, upper bound: 56.4303154
time: 0.60 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -278.9472961, 522.5424805, -120.2060547, 217.4064484, -496.3537598, 642.7485352
1: -58.4989815, 66.3320999, -23.8955574, 27.5374279, -86.0364075, 90.2276611
2: -41.0573997, 67.8399963, -17.0973358, 28.4529037, -69.5102997, 84.9373245
3: -39.0714874, 112.9678192, -16.3024616, 47.5721283, -86.6436157, 129.2702637
4: -33.8307724, 83.7769852, -13.9706097, 35.3413849, -69.1721573, 97.7475967

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4293939, upper bound: 56.4648271
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4305569, upper bound: 56.4645144
time: 0.62 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -278.9472961, 522.5424805, -276.8609009, 520.8369141, -799.7841797, 799.4032593
1: -58.4989815, 66.3320999, -58.3363228, 66.0971909, -124.5961533, 124.6684113
2: -41.0573997, 67.8399963, -40.8916054, 67.5437698, -108.6011658, 108.7315979
3: -39.0714874, 112.9678192, -38.8369293, 112.4029465, -151.4744263, 151.8047028
4: -33.8307724, 83.7769852, -33.6825638, 83.3718338, -117.2026062, 117.4595490

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4293939, upper bound: 56.4698768
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4305569, upper bound: 56.4695430
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.69 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.69
Output dim: 4, lower bound: -56.4257477, upper bound: 56.4242232
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.69
Output dim: 4, lower bound: -56.4258014, upper bound: 56.4303154
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 4, lower bound: -56.4293939, upper bound: 56.4648271
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 4, lower bound: -56.4305569, upper bound: 56.4645144
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 4, lower bound: -56.4293939, upper bound: 56.4698768
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 4, lower bound: -56.4305569, upper bound: 56.4695430

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -269.0200806, 502.0541077, -120.2060547, 217.4064484, -486.4265137, 622.2601318
1: -56.1295013, 63.7700920, -23.8955574, 27.5374279, -83.6669235, 87.6656342
2: -39.4665489, 65.2759933, -17.0973358, 28.4529037, -67.9194489, 82.3733139
3: -37.6081238, 108.7365952, -16.3024616, 47.5721283, -85.1802521, 125.0390472
4: -32.5216179, 80.6189651, -13.9706097, 35.3413849, -67.8629990, 94.5895767

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4141522, upper bound: 56.4575384
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4242398, upper bound: 56.4621123
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -266.1112061, 497.5191040, -120.2060547, 217.4064484, -483.5176392, 617.7251587
1: -55.7360458, 63.1897964, -23.8955574, 27.5374279, -83.2734680, 87.0853424
2: -39.1470680, 64.6238937, -17.0973358, 28.4529037, -67.5999527, 81.7212143
3: -37.3194618, 107.7119675, -16.3024616, 47.5721283, -84.8915863, 124.0144196
4: -32.2555542, 79.7748413, -13.9706097, 35.3413849, -67.5969391, 93.7454453

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4195386, upper bound: 56.4575802
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4254526, upper bound: 56.4619511
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -269.0200806, 502.0541077, -276.8609009, 520.8369141, -789.8569946, 778.9147339
1: -56.1295013, 63.7700920, -58.3363228, 66.0971909, -122.2266922, 122.1064072
2: -39.4665489, 65.2759933, -40.8916054, 67.5437698, -107.0103149, 106.1675873
3: -37.6081238, 108.7365952, -38.8369293, 112.4029465, -150.0110779, 147.5735168
4: -32.5216179, 80.6189651, -33.6825638, 83.3718338, -115.8934479, 114.3015289

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4681157
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4692866
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -266.1112061, 497.5191040, -276.8609009, 520.8369141, -786.9481201, 774.3797607
1: -55.7360458, 63.1897964, -58.3363228, 66.0971909, -121.8332291, 121.5261230
2: -39.1470680, 64.6238937, -40.8916054, 67.5437698, -106.6908188, 105.5154877
3: -37.3194618, 107.7119675, -38.8369293, 112.4029465, -149.7224121, 146.5488892
4: -32.2555542, 79.7748413, -33.6825638, 83.3718338, -115.6273880, 113.4574051

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692759, upper bound: 56.4683553
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692759, upper bound: 56.4695430
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.81 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4141522, upper bound: 56.4575384
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4242398, upper bound: 56.4621123
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4195386, upper bound: 56.4575802
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4254526, upper bound: 56.4619511
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4681157
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4692866
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4692759, upper bound: 56.4683553
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -56.4692759, upper bound: 56.4695430

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -269.0200806, 502.0541077, -107.9755173, 191.0806580, -460.1007385, 610.0295410
1: -56.1295013, 63.7700920, -20.8968124, 24.4081917, -80.5376892, 84.6668854
2: -39.4665489, 65.2759933, -15.1291151, 25.2170868, -64.6836395, 80.4050980
3: -37.6081238, 108.7365952, -14.4785795, 42.2450104, -79.8531342, 123.2151718
4: -32.5216179, 80.6189651, -12.3653088, 31.3789082, -63.9005280, 92.9842529

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4164794, upper bound: 56.4516275
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4166859, upper bound: 56.4526440
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -269.0200806, 502.0541077, -109.0046463, 193.9721069, -462.9921570, 611.0586548
1: -56.1295013, 63.7700920, -21.3251362, 24.6041279, -80.7336273, 85.0952072
2: -39.4665489, 65.2759933, -15.3032570, 25.4882908, -64.9548416, 80.5792389
3: -37.6081238, 108.7365952, -14.6516237, 42.7463951, -80.3545151, 123.3882217
4: -32.5216179, 80.6189651, -12.4878349, 31.6933994, -64.2150116, 93.1067963

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4224106, upper bound: 56.4496945
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4242398, upper bound: 56.4621123
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -266.1112061, 497.5191040, -107.9755173, 191.0806580, -457.1918640, 605.4945679
1: -55.7360458, 63.1897964, -20.8968124, 24.4081917, -80.1442261, 84.0866013
2: -39.1470680, 64.6238937, -15.1291151, 25.2170868, -64.3641434, 79.7529984
3: -37.3194618, 107.7119675, -14.4785795, 42.2450104, -79.5644684, 122.1905441
4: -32.2555542, 79.7748413, -12.3653088, 31.3789082, -63.6344604, 92.1401215

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4167825, upper bound: 56.4566286
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4195386, upper bound: 56.4575802
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -266.1112061, 497.5191040, -109.0046463, 193.9721069, -460.0832825, 606.5236816
1: -55.7360458, 63.1897964, -21.3251362, 24.6041279, -80.3401642, 84.5149155
2: -39.1470680, 64.6238937, -15.3032570, 25.4882908, -64.6353455, 79.9271317
3: -37.3194618, 107.7119675, -14.6516237, 42.7463951, -80.0658569, 122.3635941
4: -32.2555542, 79.7748413, -12.4878349, 31.6933994, -63.9489517, 92.2626724

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4231296, upper bound: 56.4504717
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4254526, upper bound: 56.4619511
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -269.0200806, 502.0541077, -267.1414185, 500.5254517, -769.5455322, 769.1952515
1: -56.1295013, 63.7700920, -55.9831009, 63.5585976, -119.6880798, 119.7531815
2: -39.4665489, 65.2759933, -39.3191032, 65.0084381, -104.4749908, 104.5950851
3: -37.6081238, 108.7365952, -37.3994675, 108.2293015, -145.8374329, 146.1360474
4: -32.5216179, 80.6189651, -32.3893356, 80.2518845, -112.7734909, 113.0083008

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4602872, upper bound: 56.4592909
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4687047
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -269.0200806, 502.0541077, -264.3298950, 496.0624390, -765.0825195, 766.3839111
1: -56.1295013, 63.7700920, -55.5958366, 62.9867058, -119.1162109, 119.3659210
2: -39.4665489, 65.2759933, -39.0053902, 64.3691483, -103.8356934, 104.2813873
3: -37.6081238, 108.7365952, -37.1174278, 107.2253952, -144.8335266, 145.8540192
4: -32.5216179, 80.6189651, -32.1286278, 79.4257584, -111.9473572, 112.7475891

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4602872, upper bound: 56.4607511
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4698768
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -266.1112061, 497.5191040, -267.1414185, 500.5254517, -766.6365967, 764.6602173
1: -55.7360458, 63.1897964, -55.9831009, 63.5585976, -119.2946167, 119.1728973
2: -39.1470680, 64.6238937, -39.3191032, 65.0084381, -104.1554794, 103.9429779
3: -37.3194618, 107.7119675, -37.3994675, 108.2293015, -145.5487671, 145.1113892
4: -32.2555542, 79.7748413, -32.3893356, 80.2518845, -112.5074387, 112.1641769

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4630208, upper bound: 56.4605508
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4210898, upper bound: 56.4683553
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -266.1112061, 497.5191040, -264.3298950, 496.0624390, -762.1735229, 761.8488770
1: -55.7360458, 63.1897964, -55.5958366, 62.9867058, -118.7227478, 118.7856293
2: -39.1470680, 64.6238937, -39.0053902, 64.3691483, -103.5162048, 103.6292877
3: -37.3194618, 107.7119675, -37.1174278, 107.2253952, -144.5448608, 144.8293610
4: -32.2555542, 79.7748413, -32.1286278, 79.4257584, -111.6813049, 111.9034653

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4630208, upper bound: 56.4614077
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4692759, upper bound: 56.4692223
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.79 seconds
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4164794, upper bound: 56.4516275
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4166859, upper bound: 56.4526440
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4224106, upper bound: 56.4496945
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4242398, upper bound: 56.4621123
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4167825, upper bound: 56.4566286
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4195386, upper bound: 56.4575802
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4231296, upper bound: 56.4504717
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4254526, upper bound: 56.4619511
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4602872, upper bound: 56.4592909
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4687047
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4602872, upper bound: 56.4607511
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4681235, upper bound: 56.4698768
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4630208, upper bound: 56.4605508
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4210898, upper bound: 56.4683553
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4630208, upper bound: 56.4614077
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 4, lower bound: -56.4692759, upper bound: 56.4692223

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -210.7586212, 391.6204529, -99.0252914, 172.2697296, -383.0283203, 490.6457520
1: -43.6484566, 49.7512436, -18.8003883, 22.1032448, -65.7517014, 68.5516052
2: -30.8433075, 51.1804771, -13.7312565, 22.8151169, -53.6584206, 64.9117279
3: -29.5310917, 85.0946121, -13.2015162, 38.3284569, -67.8595505, 98.2961273
4: -25.4105110, 63.1013680, -11.2066097, 28.4476242, -53.8581276, 74.3079758

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4115535, upper bound: 56.4510453
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4115535, upper bound: 56.4516275
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -265.5875854, 494.7269592, -107.4814224, 190.0902557, -455.6778564, 602.2083740
1: -55.3030548, 62.8307190, -20.7745533, 24.2900944, -79.5931473, 83.6052704
2: -38.9044304, 64.3346481, -15.0558872, 25.0942478, -63.9986801, 79.3905182
3: -37.0947609, 107.2065353, -14.4098654, 42.0373039, -79.1320648, 121.6163940
4: -32.0647430, 79.4789886, -12.3036451, 31.2317410, -63.2964821, 91.7826309

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4152002, upper bound: 56.4522566
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4152002, upper bound: 56.4526440
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -107.5185623, 190.9194489, -416.2724915, 524.1595459
1: -46.6240082, 52.9673119, -20.9628906, 24.2226429, -70.8466415, 73.9301987
2: -32.7964439, 54.2655182, -15.0549507, 25.1203003, -57.9167442, 69.3204651
3: -31.3259716, 90.6217194, -14.4177856, 42.0995293, -73.4254990, 105.0394974
4: -27.0228558, 66.9699402, -12.2876396, 31.2377701, -58.2606277, 79.2575684

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4485902
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4163410, upper bound: 56.4496945
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -109.0046463, 193.9721069, -461.5630493, 608.3172607
1: -55.8285866, 63.4212761, -21.3251362, 24.6041279, -80.4327087, 84.7464066
2: -39.2468033, 64.9129715, -15.3032570, 25.4882908, -64.7350922, 80.2162323
3: -37.3973541, 108.1530457, -14.6516237, 42.7463951, -80.1437454, 122.8046722
4: -32.3414459, 80.1706467, -12.4878349, 31.6933994, -64.0348434, 92.6584778

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4163410, upper bound: 56.4608050
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4163410, upper bound: 56.4621123
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -256.7168884, 478.7636108, -106.8419342, 188.6911926, -445.4080505, 585.6055298
1: -53.4720726, 60.8306885, -20.6146851, 24.1126251, -77.5847015, 81.4453735
2: -37.6523628, 62.2922592, -14.9499140, 24.9052620, -62.5576248, 77.2421722
3: -35.8512840, 103.7357407, -14.3130989, 41.7485962, -77.5998840, 118.0488281
4: -31.0028267, 76.9585114, -12.2168617, 31.0062828, -62.0091095, 89.1753693

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -265.7009583, 496.6644287, -107.9755173, 191.0806580, -456.7816162, 604.6398926
1: -55.6362076, 63.0820503, -20.8968124, 24.4081917, -80.0444031, 83.9788666
2: -39.0764160, 64.5190125, -15.1291151, 25.2170868, -64.2934952, 79.6481171
3: -37.2483177, 107.5315857, -14.4785795, 42.2450104, -79.4933319, 122.0101624
4: -32.1958771, 79.6516876, -12.3653088, 31.3789082, -63.5747795, 92.0169754

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4146326, upper bound: 56.4569851
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4179536, upper bound: 56.4575802
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -107.5185623, 190.9194489, -413.2910461, 519.7183838
1: -46.2298126, 52.3824730, -20.9628906, 24.2226429, -70.4524536, 73.3453598
2: -32.4683037, 53.6351471, -15.0549507, 25.1203003, -57.5886040, 68.6900940
3: -31.0229454, 89.5735931, -14.4177856, 42.0995293, -73.1224747, 103.9913788
4: -26.7529831, 66.1527557, -12.2876396, 31.2377701, -57.9907532, 78.4403992

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4210726, upper bound: 56.4493613
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4210726, upper bound: 56.4498971
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -109.0046463, 193.9721069, -458.5692749, 603.5831299
1: -55.4112587, 62.8176956, -21.3251362, 24.6041279, -80.0153809, 84.1428223
2: -38.9124107, 64.2343979, -15.3032570, 25.4882908, -64.4006882, 79.5376434
3: -37.0955200, 107.0900192, -14.6516237, 42.7463951, -79.8419189, 121.7416458
4: -32.0636482, 79.2948456, -12.4878349, 31.6933994, -63.7570419, 91.7826767

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4606453
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4616725
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -265.8039551, 497.7438660, -723.0968018, 682.4447632
1: -46.6240082, 52.9673119, -55.6768951, 63.2111511, -109.8351593, 108.6441879
2: -32.7964439, 54.2655182, -39.1048622, 64.6651306, -97.4615479, 93.3703766
3: -31.3259716, 90.6217194, -37.1991005, 107.6513062, -138.9772644, 127.8208160
4: -27.0228558, 66.9699402, -32.2155762, 79.8295593, -106.8524170, 99.1855087

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4588299, upper bound: 56.4589833
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4588299, upper bound: 56.4592909
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -267.1414185, 500.5254517, -768.1164551, 766.4538574
1: -55.8285866, 63.4212761, -55.9831009, 63.5585976, -119.3871613, 119.4043732
2: -39.2468033, 64.9129715, -39.3191032, 65.0084381, -104.2552261, 104.2320709
3: -37.3973541, 108.1530457, -37.3994675, 108.2293015, -145.6266479, 145.5525055
4: -32.3414459, 80.1706467, -32.3893356, 80.2518845, -112.5933304, 112.5599823

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4601007, upper bound: 56.4675964
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4687047
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -263.0928040, 493.3783569, -718.7312622, 679.7335815
1: -46.6240082, 52.9673119, -55.2869034, 62.6577530, -109.2817383, 108.2542114
2: -32.7964439, 54.2655182, -38.8002739, 64.0389481, -96.8353729, 93.0657959
3: -31.3259716, 90.6217194, -36.9279823, 106.6692276, -137.9951935, 127.5496902
4: -27.0228558, 66.9699402, -31.9626713, 79.0218582, -106.0447159, 98.9325943

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4599811
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4607511
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -264.3298950, 496.0624390, -763.6533813, 763.6425171
1: -55.8285866, 63.4212761, -55.5958366, 62.9867058, -118.8152924, 119.0171127
2: -39.2468033, 64.9129715, -39.0053902, 64.3691483, -103.6159439, 103.9183655
3: -37.3973541, 108.1530457, -37.1174278, 107.2253952, -144.6227417, 145.2704773
4: -32.3414459, 80.1706467, -32.1286278, 79.4257584, -111.7671967, 112.2992706

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4240776, upper bound: 56.4685942
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4613966, upper bound: 56.4698768
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -265.8039551, 497.7438660, -720.1154175, 678.0036621
1: -46.2298126, 52.3824730, -55.6768951, 63.2111511, -109.4409637, 108.0593643
2: -32.4683037, 53.6351471, -39.1048622, 64.6651306, -97.1334381, 92.7400055
3: -31.0229454, 89.5735931, -37.1991005, 107.6513062, -138.6742554, 126.7726898
4: -26.7529831, 66.1527557, -32.2155762, 79.8295593, -106.5825348, 98.3683319

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4598323, upper bound: 56.4602793
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4598323, upper bound: 56.4605508
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -267.1414185, 500.5254517, -765.1225586, 761.7197876
1: -55.4112587, 62.8176956, -55.9831009, 63.5585976, -118.9698334, 118.8007965
2: -38.9124107, 64.2343979, -39.3191032, 65.0084381, -103.9208298, 103.5534821
3: -37.0955200, 107.0900192, -37.3994675, 108.2293015, -145.3248291, 144.4894562
4: -32.0636482, 79.2948456, -32.3893356, 80.2518845, -112.3155365, 111.6841812

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4672837
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4210207, upper bound: 56.4683553
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -263.0928040, 493.3783569, -715.7498779, 675.2925415
1: -46.2298126, 52.3824730, -55.2869034, 62.6577530, -108.8875656, 107.6693726
2: -32.4683037, 53.6351471, -38.8002739, 64.0389481, -96.5072479, 92.4354248
3: -31.0229454, 89.5735931, -36.9279823, 106.6692276, -137.6921692, 126.5015640
4: -26.7529831, 66.1527557, -31.9626713, 79.0218582, -105.7748337, 98.1154251

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4611846, upper bound: 56.4611652
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4290392, upper bound: 56.4605508
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -264.3298950, 496.0624390, -760.6594849, 758.9083252
1: -55.4112587, 62.8176956, -55.5958366, 62.9867058, -118.3979645, 118.4135284
2: -38.9124107, 64.2343979, -39.0053902, 64.3691483, -103.2815475, 103.2397919
3: -37.0955200, 107.0900192, -37.1174278, 107.2253952, -144.3209076, 144.2074280
4: -32.0636482, 79.2948456, -32.1286278, 79.4257584, -111.4894104, 111.4234772

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4625350, upper bound: 56.4682143
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4252721, upper bound: 56.4692223
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.10 seconds
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4115535, upper bound: 56.4510453
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4115535, upper bound: 56.4516275
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4152002, upper bound: 56.4522566
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4152002, upper bound: 56.4526440
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4485902
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4163410, upper bound: 56.4496945
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4163410, upper bound: 56.4608050
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4163410, upper bound: 56.4621123
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4146326, upper bound: 56.4569851
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4179536, upper bound: 56.4575802
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4210726, upper bound: 56.4493613
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4210726, upper bound: 56.4498971
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4606453
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4616725
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4588299, upper bound: 56.4589833
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4588299, upper bound: 56.4592909
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4601007, upper bound: 56.4675964
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4687047
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4599811
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4607511
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4240776, upper bound: 56.4685942
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4613966, upper bound: 56.4698768
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4598323, upper bound: 56.4602793
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4598323, upper bound: 56.4605508
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4672837
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4210207, upper bound: 56.4683553
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4611846, upper bound: 56.4611652
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4290392, upper bound: 56.4605508
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4625350, upper bound: 56.4682143
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 4, lower bound: -56.4252721, upper bound: 56.4692223

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -210.7586212, 391.6204529, -92.8994370, 157.2004852, -367.9590759, 484.5198975
1: -43.6484566, 49.7512436, -17.0660973, 20.3053989, -63.9538536, 66.8173370
2: -30.8433075, 51.1804771, -12.6252213, 21.0680485, -51.9113503, 63.8056984
3: -29.5310917, 85.0946121, -12.2298841, 35.4313660, -64.9624557, 97.3244934
4: -25.4105110, 63.1013680, -10.2912464, 26.3088303, -51.7193375, 73.3926163

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4136901, upper bound: 56.4509693
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4136901, upper bound: 56.4510453
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -210.7586212, 391.6204529, -89.9481277, 153.0362854, -363.7949219, 481.5685730
1: -43.6484566, 49.7512436, -16.7086792, 19.7436752, -63.3921242, 66.4599228
2: -30.8433075, 51.1804771, -12.2971201, 20.4256973, -51.2690048, 63.4775848
3: -29.5310917, 85.0946121, -11.9287548, 34.3756256, -63.9067154, 97.0233688
4: -25.4105110, 63.1013680, -10.0335445, 25.4835358, -50.8940430, 73.1349106

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4136901, upper bound: 56.4515700
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4148841, upper bound: 56.4516275
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -265.5875854, 494.7269592, -101.1468277, 174.7703094, -440.3579102, 595.8737793
1: -55.3030548, 62.8307190, -18.9679241, 22.4685345, -77.7715836, 81.7986450
2: -38.9044304, 64.3346481, -13.9328213, 23.2670937, -62.1715202, 78.2674561
3: -37.0947609, 107.2065353, -13.4025698, 39.0380135, -76.1327744, 120.6090927
4: -32.0647430, 79.4789886, -11.3644457, 29.0294838, -61.0942192, 90.8434372

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4151999, upper bound: 56.4522566
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4121881, upper bound: 56.4515784
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -265.5875854, 494.7269592, -97.7451172, 169.4994965, -435.0870972, 592.4720459
1: -55.3030548, 62.8307190, -18.5333786, 21.7717190, -77.0747681, 81.3640900
2: -38.9044304, 64.3346481, -13.5310221, 22.4818363, -61.3862648, 77.8656693
3: -37.0947609, 107.2065353, -13.0350189, 37.7711868, -74.8659515, 120.2415466
4: -32.0647430, 79.4789886, -11.0464029, 28.0256729, -60.0904160, 90.5253906

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4121881, upper bound: 56.4526437
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4121881, upper bound: 56.4519709
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -103.1247025, 180.6422272, -405.9953003, 519.7656860
1: -46.6240082, 52.9673119, -19.7402325, 23.0165615, -69.6405487, 72.7075424
2: -32.7964439, 54.2655182, -14.3081684, 23.8860111, -56.6824532, 68.5736847
3: -31.3259716, 90.6217194, -13.7298317, 40.1205940, -71.4465637, 104.3515472
4: -27.0228558, 66.9699402, -11.6485252, 29.7695141, -56.7923698, 78.6184387

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4485902
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4157537, upper bound: 56.4404280
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -97.1439896, 170.5668182, -395.9199219, 513.7849121
1: -46.6240082, 52.9673119, -18.8132038, 21.6954842, -68.3194656, 71.7805176
2: -32.7964439, 54.2655182, -13.5354939, 22.4958477, -55.2922859, 67.8010101
3: -31.3259716, 90.6217194, -13.0277290, 37.8372078, -69.1631699, 103.6494446
4: -27.0228558, 66.9699402, -11.0323696, 28.0056362, -55.0284843, 78.0022964

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4496945
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4157537, upper bound: 56.4415413
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -104.6825104, 183.8886108, -451.4795532, 603.9951782
1: -55.8285866, 63.4212761, -20.1189842, 23.4212418, -79.2498169, 83.5402603
2: -39.2468033, 64.9129715, -14.5684948, 24.2825737, -63.5293770, 79.4814682
3: -37.3973541, 108.1530457, -13.9753847, 40.8064423, -78.2037811, 122.1284256
4: -32.3414459, 80.1706467, -11.8593607, 30.2582436, -62.5996895, 92.0300064

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4214571, upper bound: 56.4606606
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4214721, upper bound: 56.4608050
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -98.7410049, 173.9835815, -441.5745544, 598.0535889
1: -55.8285866, 63.4212761, -19.2108669, 22.1242161, -77.9527969, 82.6321335
2: -39.2468033, 64.9129715, -13.8079567, 22.9111519, -62.1579514, 78.7209320
3: -37.3973541, 108.1530457, -13.2823095, 38.5528336, -75.9501648, 121.4353485
4: -32.3414459, 80.1706467, -11.2533741, 28.5200424, -60.8614883, 91.4240189

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4214571, upper bound: 56.4619677
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4214721, upper bound: 56.4621123
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -265.7009583, 496.6644287, -101.5128250, 175.5041656, -441.2051392, 598.1772461
1: -55.6362076, 63.0820503, -19.0588589, 22.5575409, -78.1937485, 82.1409073
2: -39.0764160, 64.5190125, -13.9876070, 23.3591366, -62.4355545, 78.5066223
3: -37.2483177, 107.5315857, -13.4513960, 39.1907310, -76.4390411, 120.9829788
4: -32.1958771, 79.6516876, -11.4093437, 29.1383343, -61.3342133, 91.0610352

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4148841, upper bound: 56.4529811
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4168211, upper bound: 56.4529392
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -265.7009583, 496.6644287, -98.1992722, 170.4810486, -436.1820068, 594.8637085
1: -55.6362076, 63.0820503, -18.6528606, 21.8924599, -77.5286713, 81.7349091
2: -39.0764160, 64.5190125, -13.6042700, 22.6031761, -61.6795883, 78.1232758
3: -37.2483177, 107.5315857, -13.0987883, 37.9729919, -75.2213135, 120.6303711
4: -32.1958771, 79.6516876, -11.1056051, 28.1750202, -60.3708954, 90.7572937

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4170300, upper bound: 56.4534543
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4139375, upper bound: 56.4532804
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -103.1247025, 180.6422272, -403.0139160, 515.3245850
1: -46.2298126, 52.3824730, -19.7402325, 23.0165615, -69.2463760, 72.1227036
2: -32.4683037, 53.6351471, -14.3081684, 23.8860111, -56.3543167, 67.9433136
3: -31.0229454, 89.5735931, -13.7298317, 40.1205940, -71.1435394, 103.3034210
4: -26.7529831, 66.1527557, -11.6485252, 29.7695141, -56.5224991, 77.8012695

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4207955, upper bound: 56.4493613
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -97.1439896, 170.5668182, -392.9385071, 509.3438416
1: -46.2298126, 52.3824730, -18.8132038, 21.6954842, -67.9252853, 71.1956787
2: -32.4683037, 53.6351471, -13.5354939, 22.4958477, -54.9641495, 67.1706390
3: -31.0229454, 89.5735931, -13.0277290, 37.8372078, -68.8601532, 102.6013184
4: -26.7529831, 66.1527557, -11.0323696, 28.0056362, -54.7586174, 77.1851273

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -104.6825104, 183.8886108, -448.4858093, 599.2609863
1: -55.4112587, 62.8176956, -20.1189842, 23.4212418, -78.8324890, 82.9366760
2: -38.9124107, 64.2343979, -14.5684948, 24.2825737, -63.1949844, 78.8028870
3: -37.0955200, 107.0900192, -13.9753847, 40.8064423, -77.9019623, 121.0654068
4: -32.0636482, 79.2948456, -11.8593607, 30.2582436, -62.3218918, 91.1542053

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4606453
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4210285, upper bound: 56.4606453
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -98.7410049, 173.9835815, -438.5807495, 593.3194580
1: -55.4112587, 62.8176956, -19.2108669, 22.1242161, -77.5354691, 82.0285568
2: -38.9124107, 64.2343979, -13.8079567, 22.9111519, -61.8235588, 78.0423431
3: -37.0955200, 107.0900192, -13.2823095, 38.5528336, -75.6483307, 120.3723297
4: -32.0636482, 79.2948456, -11.2533741, 28.5200424, -60.5836906, 90.5482178

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4616644
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4227711, upper bound: 56.4616725
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -224.1501465, 415.6130676, -640.9661255, 640.7910767
1: -46.6240082, 52.9673119, -46.5260315, 52.8243332, -99.4483414, 99.4933395
2: -32.7964439, 54.2655182, -32.6965485, 54.0849915, -86.8814087, 86.9620667
3: -31.3259716, 90.6217194, -31.1839104, 90.2783737, -121.6043243, 121.8056335
4: -27.0228558, 66.9699402, -26.9336452, 66.7248306, -93.7476883, 93.9035797

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4192669, upper bound: 56.4581495
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4494006
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -265.7260132, 497.7924500, -723.1454468, 682.3669434
1: -46.6240082, 52.9673119, -55.6827927, 63.2111206, -109.8351212, 108.6500854
2: -32.7964439, 54.2655182, -39.1002007, 64.6471558, -97.4435730, 93.3657227
3: -31.3259716, 90.6217194, -37.1901855, 107.6487579, -138.9746857, 127.8119049
4: -27.0228558, 66.9699402, -32.2101021, 79.8059006, -106.8287582, 99.1800385

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4192669, upper bound: 56.4592909
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4494010
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -224.1501465, 415.6130676, -683.2040405, 723.4627686
1: -55.8285866, 63.4212761, -46.5260315, 52.8243332, -108.6529236, 109.9473114
2: -39.2468033, 64.9129715, -32.6965485, 54.0849915, -93.3317719, 97.6095200
3: -37.3973541, 108.1530457, -31.1839104, 90.2783737, -127.6757278, 139.3369598
4: -32.3414459, 80.1706467, -26.9336452, 66.7248306, -99.0662689, 107.1042938

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600467, upper bound: 56.4666011
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600704, upper bound: 56.4675536
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -265.7260132, 497.7924500, -765.3834229, 765.0386963
1: -55.8285866, 63.4212761, -55.6827927, 63.2111206, -119.0397034, 119.1040573
2: -39.2468033, 64.9129715, -39.1002007, 64.6471558, -103.8939362, 104.0131683
3: -37.3973541, 108.1530457, -37.1901855, 107.6487579, -145.0460968, 145.3432312
4: -32.3414459, 80.1706467, -32.2101021, 79.8059006, -112.1473465, 112.3807526

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600467, upper bound: 56.4677216
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4600704, upper bound: 56.4685953
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -221.1691437, 411.1789246, -636.5318604, 637.8100586
1: -46.6240082, 52.9673119, -46.1322098, 52.2386818, -98.8626862, 99.0994873
2: -32.7964439, 54.2655182, -32.3670921, 53.4560623, -86.2525024, 86.6326141
3: -31.3259716, 90.6217194, -30.8784237, 89.2303925, -120.5563431, 121.5001450
4: -27.0228558, 66.9699402, -26.6626072, 65.9110031, -92.9338608, 93.6325302

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4599811
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4494006, upper bound: 56.4504553
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -225.3530884, 416.6409607, -262.8239136, 493.1330566, -718.4860229, 679.4647827
1: -46.6240082, 52.9673119, -55.2722130, 62.6158485, -109.2398376, 108.2395096
2: -32.7964439, 54.2655182, -38.7716637, 63.9805908, -96.7770157, 93.0371857
3: -31.3259716, 90.6217194, -36.8949852, 106.6060486, -137.9319916, 127.5166855
4: -27.0228558, 66.9699402, -31.9376850, 78.9470291, -105.9698868, 98.9076233

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4607511
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4516129, upper bound: 56.4509211
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -221.1691437, 411.1789246, -678.7698364, 720.4818115
1: -55.8285866, 63.4212761, -46.1322098, 52.2386818, -108.0672684, 109.5534592
2: -39.2468033, 64.9129715, -32.3670921, 53.4560623, -92.7028656, 97.2800598
3: -37.3973541, 108.1530457, -30.8784237, 89.2303925, -126.6277390, 139.0314636
4: -32.3414459, 80.1706467, -26.6626072, 65.9110031, -98.2524490, 106.8332443

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4613427, upper bound: 56.4676000
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4613664, upper bound: 56.4685525
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -267.5909729, 499.3126526, -262.8239136, 493.1330566, -760.7239990, 762.1365356
1: -55.8285866, 63.4212761, -55.2722130, 62.6158485, -118.4444275, 118.6934814
2: -39.2468033, 64.9129715, -38.7716637, 63.9805908, -103.2273865, 103.6846313
3: -37.3973541, 108.1530457, -36.8949852, 106.6060486, -144.0033875, 145.0480347
4: -32.3414459, 80.1706467, -31.9376850, 78.9470291, -111.2884750, 112.1083298

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4613427, upper bound: 56.4688089
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4613664, upper bound: 56.4695603
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -224.1501465, 415.6130676, -637.9847412, 636.3499756
1: -46.2298126, 52.3824730, -46.5260315, 52.8243332, -99.0541458, 98.9085083
2: -32.4683037, 53.6351471, -32.6965485, 54.0849915, -86.5532990, 86.3316956
3: -31.0229454, 89.5735931, -31.1839104, 90.2783737, -121.3013153, 120.7575073
4: -26.7529831, 66.1527557, -26.9336452, 66.7248306, -93.4778061, 93.0864029

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -265.7260132, 497.7924500, -720.1641235, 677.9259033
1: -46.2298126, 52.3824730, -55.6827927, 63.2111206, -109.4409332, 108.0652542
2: -32.4683037, 53.6351471, -39.1002007, 64.6471558, -97.1154633, 92.7353516
3: -31.0229454, 89.5735931, -37.1901855, 107.6487579, -138.6716919, 126.7637787
4: -26.7529831, 66.1527557, -32.2101021, 79.8059006, -106.5588760, 98.3628540

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4507399, upper bound: 56.4605115
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4512322, upper bound: 56.4594455
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -224.1501465, 415.6130676, -680.2102051, 718.7286377
1: -55.4112587, 62.8176956, -46.5260315, 52.8243332, -108.2355957, 109.3437271
2: -38.9124107, 64.2343979, -32.6965485, 54.0849915, -92.9973831, 96.9309464
3: -37.0955200, 107.0900192, -31.1839104, 90.2783737, -127.3738937, 138.2739258
4: -32.0636482, 79.2948456, -26.9336452, 66.7248306, -98.7884827, 106.2284927

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4611584, upper bound: 56.4667371
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4612203, upper bound: 56.4672408
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -265.7260132, 497.7924500, -762.3895264, 760.3044434
1: -55.4112587, 62.8176956, -55.6827927, 63.2111206, -118.6223755, 118.5004730
2: -38.9124107, 64.2343979, -39.1002007, 64.6471558, -103.5595474, 103.3345871
3: -37.0955200, 107.0900192, -37.1901855, 107.6487579, -144.7442474, 144.2802124
4: -32.0636482, 79.2948456, -32.2101021, 79.8059006, -111.8695526, 111.5049438

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4529402, upper bound: 56.4638539
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4608380, upper bound: 56.4681728
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -221.1691437, 411.1789246, -633.5505371, 633.3689575
1: -46.2298126, 52.3824730, -46.1322098, 52.2386818, -98.4684906, 98.5146637
2: -32.4683037, 53.6351471, -32.3670921, 53.4560623, -85.9243622, 86.0022430
3: -31.0229454, 89.5735931, -30.8784237, 89.2303925, -120.2533417, 120.4520187
4: -26.7529831, 66.1527557, -26.6626072, 65.9110031, -92.6639862, 92.8153610

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4607228, upper bound: 56.4608003
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4611846, upper bound: 56.4611652
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -222.3716888, 412.1998596, -262.8239136, 493.1330566, -715.5046997, 675.0237427
1: -46.2298126, 52.3824730, -55.2722130, 62.6158485, -108.8456573, 107.6546860
2: -32.4683037, 53.6351471, -38.7716637, 63.9805908, -96.4488983, 92.4068146
3: -31.0229454, 89.5735931, -36.8949852, 106.6060486, -137.6289978, 126.4685669
4: -26.7529831, 66.1527557, -31.9376850, 78.9470291, -105.7000122, 98.0904388

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4607228, upper bound: 56.4611708
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4611846, upper bound: 56.4614077
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -221.1691437, 411.1789246, -675.7759399, 715.7476196
1: -55.4112587, 62.8176956, -46.1322098, 52.2386818, -107.6499405, 108.9498825
2: -38.9124107, 64.2343979, -32.3670921, 53.4560623, -92.3684692, 96.6014862
3: -37.0955200, 107.0900192, -30.8784237, 89.2303925, -126.3259048, 137.9684448
4: -32.0636482, 79.2948456, -26.6626072, 65.9110031, -97.9746552, 105.9574509

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4625019, upper bound: 56.4675991
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4625338, upper bound: 56.4681726
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -264.5971985, 494.5784607, -262.8239136, 493.1330566, -757.7301025, 757.4023438
1: -55.4112587, 62.8176956, -55.2722130, 62.6158485, -118.0270920, 118.0899048
2: -38.9124107, 64.2343979, -38.7716637, 63.9805908, -102.8929977, 103.0060577
3: -37.0955200, 107.0900192, -36.8949852, 106.6060486, -143.7015533, 143.9850006
4: -32.0636482, 79.2948456, -31.9376850, 78.9470291, -111.0106812, 111.2325287

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4625019, upper bound: 56.4686624
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4625338, upper bound: 56.4692200
time: 0.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.25 seconds
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4136901, upper bound: 56.4509693
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4136901, upper bound: 56.4510453
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4136901, upper bound: 56.4515700
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4148841, upper bound: 56.4516275
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4151999, upper bound: 56.4522566
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4121881, upper bound: 56.4515784
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4121881, upper bound: 56.4526437
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4121881, upper bound: 56.4519709
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4485902
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4157537, upper bound: 56.4404280
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4496945
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4157537, upper bound: 56.4415413
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4214571, upper bound: 56.4606606
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4214721, upper bound: 56.4608050
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4214571, upper bound: 56.4619677
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4214721, upper bound: 56.4621123
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4148841, upper bound: 56.4529811
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4168211, upper bound: 56.4529392
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4170300, upper bound: 56.4534543
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4139375, upper bound: 56.4532804
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4606453
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4210285, upper bound: 56.4606453
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4616644
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4227711, upper bound: 56.4616725
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4192669, upper bound: 56.4581495
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4494006
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4192669, upper bound: 56.4592909
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4198263, upper bound: 56.4494010
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4600467, upper bound: 56.4666011
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4600704, upper bound: 56.4675536
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4600467, upper bound: 56.4677216
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4600704, upper bound: 56.4685953
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4599811
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4494006, upper bound: 56.4504553
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4601652, upper bound: 56.4607511
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4516129, upper bound: 56.4509211
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4613427, upper bound: 56.4676000
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4613664, upper bound: 56.4685525
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4613427, upper bound: 56.4688089
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4613664, upper bound: 56.4695603
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4507399, upper bound: 56.4605115
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4512322, upper bound: 56.4594455
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4611584, upper bound: 56.4667371
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4612203, upper bound: 56.4672408
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4529402, upper bound: 56.4638539
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4608380, upper bound: 56.4681728
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4607228, upper bound: 56.4608003
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4611846, upper bound: 56.4611652
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4607228, upper bound: 56.4611708
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4611846, upper bound: 56.4614077
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4625019, upper bound: 56.4675991
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4625338, upper bound: 56.4681726
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4625019, upper bound: 56.4686624
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 4, lower bound: -56.4625338, upper bound: 56.4692200

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -210.3384552, 390.8152771, -92.8994370, 157.2004852, -367.5389404, 483.7147217
1: -43.5528450, 49.6501007, -17.0660973, 20.3053989, -63.8582382, 66.7162018
2: -30.7793541, 51.0804825, -12.6252213, 21.0680485, -51.8473969, 63.7057037
3: -29.4665451, 84.9222031, -12.2298841, 35.4313660, -64.8979034, 97.1520767
4: -25.3562679, 62.9815025, -10.2912464, 26.3088303, -51.6650963, 73.2727509

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4124434, upper bound: 56.4490505
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4124434, upper bound: 56.4509693
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -210.5222778, 391.1582642, -92.8994370, 157.2004852, -367.7227783, 484.0577087
1: -43.5959435, 49.6924667, -17.0660973, 20.3053989, -63.9013405, 66.7585602
2: -30.8069553, 51.1208420, -12.6252213, 21.0680485, -51.8749924, 63.7460632
3: -29.4958496, 84.9965210, -12.2298841, 35.4313660, -64.9272079, 97.2264023
4: -25.3801479, 63.0285645, -10.2912464, 26.3088303, -51.6889687, 73.3198090

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4073299, upper bound: 56.4492100
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4073299, upper bound: 56.4510453
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -210.3384552, 390.8152771, -89.9481277, 153.0362854, -363.3747559, 480.7633972
1: -43.5528450, 49.6501007, -16.7086792, 19.7436752, -63.2965088, 66.3587799
2: -30.7793541, 51.0804825, -12.2971201, 20.4256973, -51.2050514, 63.3776016
3: -29.4665451, 84.9222031, -11.9287548, 34.3756256, -63.8421707, 96.8509521
4: -25.3562679, 62.9815025, -10.0335445, 25.4835358, -50.8398018, 73.0150452

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -210.5222778, 391.1582642, -89.9481277, 153.0362854, -363.5585632, 481.1063843
1: -43.5959435, 49.6924667, -16.7086792, 19.7436752, -63.3396187, 66.4011459
2: -30.8069553, 51.1208420, -12.2971201, 20.4256973, -51.2326508, 63.4179573
3: -29.4958496, 84.9965210, -11.9287548, 34.3756256, -63.8714638, 96.9252777
4: -25.3801479, 63.0285645, -10.0335445, 25.4835358, -50.8636780, 73.0621109

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -265.7788086, 495.3089294, -101.1468277, 174.7703094, -440.5491028, 596.4557495
1: -55.3735275, 62.8973618, -18.9679241, 22.4685345, -77.8420563, 81.8652878
2: -38.9412537, 64.3983765, -13.9328213, 23.2670937, -62.2083435, 78.3311996
3: -37.1190147, 107.3111496, -13.4025698, 39.0380135, -76.1570282, 120.7137070
4: -32.0910225, 79.5582123, -11.3644457, 29.0294838, -61.1204948, 90.9226608

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4124434, upper bound: 56.4502982
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4080159, upper bound: 56.4515784
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -265.3187561, 494.1998291, -101.1468277, 174.7703094, -440.0889893, 595.3466797
1: -55.2420883, 62.7640533, -18.9679241, 22.4685345, -77.7106018, 81.7319717
2: -38.8631592, 64.2670746, -13.9328213, 23.2670937, -62.1302490, 78.1998901
3: -37.0557175, 107.0951385, -13.4025698, 39.0380135, -76.0937347, 120.4976959
4: -32.0303688, 79.3959198, -11.3644457, 29.0294838, -61.0598488, 90.7603683

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4085385, upper bound: 56.4502982
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4126908, upper bound: 56.4515784
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -265.7788086, 495.3089294, -97.7451172, 169.4994965, -435.2783203, 593.0540771
1: -55.3735275, 62.8973618, -18.5333786, 21.7717190, -77.1452332, 81.4307404
2: -38.9412537, 64.3983765, -13.5310221, 22.4818363, -61.4230881, 77.9293976
3: -37.1190147, 107.3111496, -13.0350189, 37.7711868, -74.8901978, 120.3461609
4: -32.0910225, 79.5582123, -11.0464029, 28.0256729, -60.1166954, 90.6046066

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -265.3187561, 494.1998291, -97.7451172, 169.4994965, -434.8181763, 591.9449463
1: -55.2420883, 62.7640533, -18.5333786, 21.7717190, -77.0137863, 81.2974243
2: -38.8631592, 64.2670746, -13.5310221, 22.4818363, -61.3449936, 77.7980957
3: -37.0557175, 107.0951385, -13.0350189, 37.7711868, -74.8269043, 120.1301422
4: -32.0303688, 79.3959198, -11.0464029, 28.0256729, -60.0560417, 90.4423141

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -225.6607666, 417.5713196, -103.1247025, 180.6422272, -406.3029785, 520.6960449
1: -46.7165413, 53.0760155, -19.7402325, 23.0165615, -69.7330933, 72.8162460
2: -32.8573380, 54.3693123, -14.3081684, 23.8860111, -56.7433472, 68.6774750
3: -31.3710880, 90.7928925, -13.7298317, 40.1205940, -71.4916840, 104.5227203
4: -27.0688438, 67.1066360, -11.6485252, 29.7695141, -56.8383560, 78.7551575

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4157974, upper bound: 56.4485902
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4157974, upper bound: 56.4485902
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -225.1204071, 416.1901245, -103.1247025, 180.6422272, -405.7626038, 519.3148193
1: -46.5717010, 52.9104042, -19.7402325, 23.0165615, -69.5882492, 72.6506348
2: -32.7608032, 54.2076073, -14.3081684, 23.8860111, -56.6468124, 68.5157776
3: -31.2914314, 90.5253830, -13.7298317, 40.1205940, -71.4120255, 104.2552185
4: -26.9930058, 66.8983078, -11.6485252, 29.7695141, -56.7625198, 78.5468292

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4195317, upper bound: 56.4404280
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4157974, upper bound: 56.4404280
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -225.6607666, 417.5713196, -97.1439896, 170.5668182, -396.2276001, 514.7152710
1: -46.7165413, 53.0760155, -18.8132038, 21.6954842, -68.4120102, 71.8892212
2: -32.8573380, 54.3693123, -13.5354939, 22.4958477, -55.3531876, 67.9048080
3: -31.3710880, 90.7928925, -13.0277290, 37.8372078, -69.2082977, 103.8206177
4: -27.0688438, 67.1066360, -11.0323696, 28.0056362, -55.0744781, 78.1390076

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4177363, upper bound: 56.4496645
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4203628, upper bound: 56.4496945
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -225.1204071, 416.1901245, -97.1439896, 170.5668182, -395.6871948, 513.3341064
1: -46.5717010, 52.9104042, -18.8132038, 21.6954842, -68.2671661, 71.7236023
2: -32.7608032, 54.2076073, -13.5354939, 22.4958477, -55.2566490, 67.7431030
3: -31.2914314, 90.5253830, -13.0277290, 37.8372078, -69.1286392, 103.5531158
4: -26.9930058, 66.8983078, -11.0323696, 28.0056362, -54.9986382, 77.9306793

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4194896, upper bound: 56.4415122
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4194896, upper bound: 56.4415413
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -258.1553650, 480.4909363, -103.8145065, 182.1802673, -440.3356018, 584.3054199
1: -53.5597420, 61.0363388, -19.9208698, 23.2072773, -76.7670212, 80.9572067
2: -37.7362366, 62.5766068, -14.4379463, 24.0631599, -61.7993927, 77.0145569
3: -35.9093742, 104.1696472, -13.8513117, 40.4496231, -76.3589859, 118.0209503
4: -31.0755901, 77.3515244, -11.7510481, 29.9931850, -61.0687752, 89.1025696

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4189396, upper bound: 56.4599630
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4214571, upper bound: 56.4606606
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -267.1218262, 498.3845825, -104.6825104, 183.8886108, -451.0104065, 603.0670776
1: -55.7263603, 63.3030777, -20.1189842, 23.4212418, -79.1475983, 83.4220581
2: -39.1702309, 64.7981186, -14.5684948, 24.2825737, -63.4528046, 79.3666000
3: -37.3202629, 107.9551773, -13.9753847, 40.8064423, -78.1266937, 121.9305649
4: -32.2771378, 80.0329208, -11.8593607, 30.2582436, -62.5353813, 91.8922806

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4052179, upper bound: 56.4533362
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4196945, upper bound: 56.4546328
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -258.1553650, 480.4909363, -97.8077316, 172.1741333, -430.3294983, 578.2986450
1: -53.5597420, 61.0363388, -19.0049305, 21.8958702, -75.4556122, 80.0412674
2: -37.7362366, 62.5766068, -13.6680536, 22.6811485, -60.4173775, 76.2446594
3: -35.9093742, 104.1696472, -13.1505337, 38.1715965, -74.0809631, 117.3201675
4: -31.0755901, 77.3515244, -11.1376257, 28.2367306, -59.3123207, 88.4891510

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4241985, upper bound: 56.4617581
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4241985, upper bound: 56.4619677
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -267.1218262, 498.3845825, -98.7410049, 173.9835815, -441.1053467, 597.1254272
1: -55.7263603, 63.3030777, -19.2108669, 22.1242161, -77.8505783, 82.5139389
2: -39.1702309, 64.7981186, -13.8079567, 22.9111519, -62.0813751, 78.6060562
3: -37.3202629, 107.9551773, -13.2823095, 38.5528336, -75.8730698, 121.2374878
4: -32.2771378, 80.0329208, -11.2533741, 28.5200424, -60.7971802, 91.2862930

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4119890, upper bound: 56.4556696
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4223230, upper bound: 56.4557693
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -211.2011719, 394.0278625, -92.8994370, 157.2004852, -368.4016418, 486.9273071
1: -44.0293388, 50.0219765, -17.0660973, 20.3053989, -64.3347321, 67.0880737
2: -31.0275192, 51.4286575, -12.6252213, 21.0680485, -52.0955582, 64.0538712
3: -29.6670609, 85.4981232, -12.2298841, 35.4313660, -65.0984268, 97.7280045
4: -25.5450649, 63.3621597, -10.2912464, 26.3088303, -51.8538895, 73.6534042

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4144873, upper bound: 56.4515175
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4170300, upper bound: 56.4529811
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -261.8643494, 488.0812683, -101.1468277, 174.7703094, -436.6346436, 589.2280884
1: -54.6343536, 62.0166473, -18.9679241, 22.4685345, -77.1028900, 80.9845734
2: -38.4276505, 63.4374390, -13.9328213, 23.2670937, -61.6947403, 77.3702469
3: -36.6539764, 105.7617416, -13.4025698, 39.0380135, -75.6919861, 119.1642914
4: -31.6666145, 78.3520660, -11.3644457, 29.0294838, -60.6960907, 89.7165146

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4140289, upper bound: 56.4518554
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4140289, upper bound: 56.4529392
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -211.2011719, 394.0278625, -89.9481277, 153.0362854, -364.2374573, 483.9759827
1: -44.0293388, 50.0219765, -16.7086792, 19.7436752, -63.7730064, 66.7306519
2: -31.0275192, 51.4286575, -12.2971201, 20.4256973, -51.4532127, 63.7257729
3: -29.6670609, 85.4981232, -11.9287548, 34.3756256, -64.0426865, 97.4268799
4: -25.5450649, 63.3621597, -10.0335445, 25.4835358, -51.0285988, 73.3957062

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -261.8643494, 488.0812683, -97.7451172, 169.4994965, -431.3638306, 585.8263550
1: -54.6343536, 62.0166473, -18.5333786, 21.7717190, -76.4060745, 80.5500183
2: -38.4276505, 63.4374390, -13.5310221, 22.4818363, -60.9094849, 76.9684601
3: -36.6539764, 105.7617416, -13.0350189, 37.7711868, -74.4251633, 118.7967453
4: -31.6666145, 78.3520660, -11.0464029, 28.0256729, -59.6922874, 89.3984680

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -255.2042389, 475.8162231, -103.8145065, 182.1802673, -437.3844910, 579.6307373
1: -53.1471367, 60.4571075, -19.9208698, 23.2072773, -76.3544159, 80.3779755
2: -37.4119720, 61.9053688, -14.4379463, 24.0631599, -61.4751320, 76.3432922
3: -35.6196022, 103.1136856, -13.8513117, 40.4496231, -76.0692291, 116.9649963
4: -30.8061352, 76.4811935, -11.7510481, 29.9931850, -60.7993202, 88.2322388

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4181917, upper bound: 56.4386459
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4606453
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4228117, upper bound: 56.4606453
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -264.1883850, 493.7238159, -104.6825104, 183.8886108, -448.0769653, 598.4063110
1: -55.3113060, 62.7100563, -20.1189842, 23.4212418, -78.7325363, 82.8290405
2: -38.8418579, 64.1296539, -14.5684948, 24.2825737, -63.1244316, 78.6981506
3: -37.0245972, 106.9096451, -13.9753847, 40.8064423, -77.8310394, 120.8850327
4: -32.0040779, 79.1718674, -11.8593607, 30.2582436, -62.2623215, 91.0312271

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4108129, upper bound: 56.4537037
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4210399, upper bound: 56.4549994
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -255.2042389, 475.8162231, -97.8077316, 172.1741333, -427.3783569, 573.6239624
1: -53.1471367, 60.4571075, -19.0049305, 21.8958702, -75.0430069, 79.4620361
2: -37.4119720, 61.9053688, -13.6680536, 22.6811485, -60.0931206, 75.5734100
3: -35.6196022, 103.1136856, -13.1505337, 38.1715965, -73.7911911, 116.2642136
4: -30.8061352, 76.4811935, -11.1376257, 28.2367306, -59.0428658, 87.6188202

Time for backsubstitution: 2.47 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.09 + 416.30 = 420.39 seconds
