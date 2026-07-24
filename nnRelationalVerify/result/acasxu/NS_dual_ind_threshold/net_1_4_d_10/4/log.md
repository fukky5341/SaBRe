## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 96.5219627187


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443)
1: (-41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358)
2: (-42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923)
3: (-48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496)
4: (-45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.61 + 1.82 = 2.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -96.6185813, upper bound: 96.6185813

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5312881, upper bound: 96.5983467
time: 0.93 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6152408, upper bound: 96.6152409
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 4, lower bound: -96.5312881, upper bound: 96.5983467
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 4, lower bound: -96.6152408, upper bound: 96.6152409

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -23.6491947, 44.5811996, -36.3742218, 60.3238182, -83.9729996, 80.9554214
1: -25.9020424, 36.9293671, -39.6944351, 51.9614906, -77.8635101, 76.6238022
2: -26.6062851, 36.6187668, -40.6514015, 51.8626137, -78.4689026, 77.2701416
3: -30.8861237, 42.7392807, -46.6203918, 60.4010353, -91.2871475, 89.3596725
4: -29.5871983, 41.9198265, -43.1367874, 60.1164207, -89.7036209, 85.0566101

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5180541
time: 0.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5983467
time: 0.57 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -47.7588348, 79.9911652, -37.5084381, 61.7592354, -109.5180664, 117.4995804
1: -52.1639366, 67.9121017, -40.9360428, 53.4188156, -105.5827484, 108.8481369
2: -53.4510193, 68.0761642, -41.9247131, 53.3292389, -106.7802582, 110.0008774
3: -61.4133224, 78.7076035, -48.0760803, 62.0458717, -123.4591751, 126.7836761
4: -56.2418251, 79.0030441, -44.4276314, 61.8733482, -118.1151733, 123.4306641

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5983467, upper bound: 96.5312881
time: 0.81 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5983467, upper bound: 96.5312881
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.18 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.18
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5180541
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5983467
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 4, lower bound: -96.5983467, upper bound: 96.5312881
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 4, lower bound: -96.5983467, upper bound: 96.5312881

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -23.6491947, 44.5811996, -47.7588348, 79.9911652, -103.6403580, 92.3400269
1: -25.9020424, 36.9293671, -52.1639366, 67.9121017, -93.8141251, 89.0933075
2: -26.6062851, 36.6187668, -53.4510193, 68.0761642, -94.6824493, 90.0697861
3: -30.8861237, 42.7392807, -61.4133224, 78.7076035, -109.5937271, 104.1525955
4: -29.5871983, 41.9198265, -56.2418251, 79.0030441, -108.5902405, 98.1616516

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4913151, upper bound: 96.5064360
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5178728, upper bound: 96.5178728
time: 0.78 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -47.7588348, 79.9911652, -23.6491947, 44.5811996, -92.3400269, 103.6403580
1: -52.1639366, 67.9121017, -25.9020424, 36.9293671, -89.0933075, 93.8141251
2: -53.4510193, 68.0761642, -26.6062851, 36.6187668, -90.0697784, 94.6824493
3: -61.4133224, 78.7076035, -30.8861237, 42.7392807, -104.1525955, 109.5937271
4: -56.2418251, 79.0030441, -29.5871983, 41.9198265, -98.1616516, 108.5902405

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5537338, upper bound: 96.5160201
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5974714, upper bound: 96.5312687
time: 1.09 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -47.7588348, 79.9911652, -47.7588348, 79.9911652, -127.7499771, 127.7499771
1: -52.1639366, 67.9121017, -52.1639366, 67.9121017, -120.0760345, 120.0760345
2: -53.4510193, 68.0761642, -53.4510193, 68.0761642, -121.5271835, 121.5271835
3: -61.4133224, 78.7076035, -61.4133224, 78.7076035, -140.1209106, 140.1208954
4: -56.2418251, 79.0030441, -56.2418251, 79.0030441, -135.2448730, 135.2448730

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5537340, upper bound: 96.5160201
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5974716, upper bound: 96.6152112
time: 0.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.57 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 4, lower bound: -96.4913151, upper bound: 96.5064360
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 4, lower bound: -96.5178728, upper bound: 96.5178728
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 4, lower bound: -96.5537338, upper bound: 96.5160201
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 4, lower bound: -96.5974714, upper bound: 96.5312687
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 4, lower bound: -96.5537340, upper bound: 96.5160201
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 4, lower bound: -96.5974716, upper bound: 96.6152112

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.6879997, 71.5514069, -22.9128723, 43.7618332, -85.4498291, 94.4642792
1: -45.5929909, 60.5228043, -25.1255646, 36.1625862, -81.7555771, 85.6483688
2: -46.7689133, 60.5786438, -25.8125191, 35.8409081, -82.6098175, 86.3911514
3: -53.8531685, 70.0829163, -30.0276375, 41.8504601, -95.7036209, 100.1105423
4: -49.6292267, 70.1049042, -28.8787460, 41.0023079, -90.6315308, 98.9836502

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4959281, upper bound: 96.4494716
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5516491, upper bound: 96.5037198
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -47.0677299, 79.0901794, -23.6332855, 44.5619965, -91.6297150, 102.7234650
1: -51.4176712, 67.0940704, -25.8851109, 36.9115639, -88.3292389, 92.9791794
2: -52.6958504, 67.2510605, -26.5890160, 36.6008987, -89.2967529, 93.8400726
3: -60.5642242, 77.7514114, -30.8672028, 42.7183990, -103.2826157, 108.6185989
4: -55.4830666, 78.0209122, -29.5707474, 41.8986435, -97.3817139, 107.5916595

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5433651, upper bound: 96.4663230
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5954258, upper bound: 96.5186601
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.6879997, 71.5514069, -46.7890282, 78.8122406, -120.5002441, 118.3404388
1: -45.5929909, 60.5228043, -51.1141739, 66.7898865, -112.3828583, 111.6369781
2: -46.7689133, 60.5786438, -52.3861618, 66.9377136, -113.7066269, 112.9647980
3: -53.8531685, 70.0829163, -60.2107544, 77.3904114, -131.2435608, 130.2936707
4: -49.6292267, 70.1049042, -55.1763687, 77.6357269, -127.2649536, 125.2812729

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5551778
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5908556
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.0677299, 79.0901794, -47.7411880, 79.9689255, -127.0366440, 126.8313675
1: -51.4176712, 67.0940704, -52.1450233, 67.8917389, -119.3094101, 119.2390900
2: -52.6958504, 67.2510605, -53.4318619, 68.0556641, -120.7515030, 120.6829224
3: -60.5642242, 77.7514114, -61.3919754, 78.6837616, -139.2479858, 139.1433868
4: -55.4830666, 78.0209122, -56.2225838, 78.9786224, -134.4616852, 134.2434998

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5945826, upper bound: 96.5661277
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5945826, upper bound: 96.5661277
time: 1.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.70 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.4959281, upper bound: 96.4494716
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5516491, upper bound: 96.5037198
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5433651, upper bound: 96.4663230
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5954258, upper bound: 96.5186601
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5551778
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5908556
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5945826, upper bound: 96.5661277
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 4, lower bound: -96.5945826, upper bound: 96.5661277

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -41.6879997, 71.5514069, -22.1488571, 42.7812309, -84.4692307, 93.7002640
1: -45.5929909, 60.5228043, -24.2961674, 35.2251091, -80.8181000, 84.8189697
2: -46.7689133, 60.5786438, -24.9756222, 34.9029617, -81.6718750, 85.5542603
3: -53.8531685, 70.0829163, -29.0839233, 40.7468796, -94.6000443, 99.1668396
4: -49.6292267, 70.1049042, -27.9935036, 39.8805008, -89.5097275, 98.0984039

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5110028, upper bound: 96.4902390
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5009241, upper bound: 96.4531550
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5489667, upper bound: 96.5037198
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5148025, upper bound: 96.4601920
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5344041, upper bound: 96.4928805
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5474053, upper bound: 96.4953539
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.1660004, 77.9299164, -18.7637291, 38.0417938, -84.2077866, 96.6936417
1: -50.4282837, 65.9576721, -20.6679897, 31.1780128, -81.6062927, 86.6256561
2: -51.6897850, 66.1108246, -21.2358284, 30.8240910, -82.5138779, 87.3466492
3: -59.4161034, 76.4025269, -24.9217834, 36.0716019, -95.4877014, 101.3242950
4: -54.4324265, 76.6484604, -24.4355717, 35.1456490, -89.5780792, 101.0840302

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5427896, upper bound: 96.4641733
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5433651, upper bound: 96.4663230
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -47.0677299, 79.0901794, -22.8273449, 43.5579681, -90.6256790, 101.9175262
1: -51.4176712, 67.0940704, -25.0134792, 35.9514008, -87.3690720, 92.1075516
2: -52.6958504, 67.2510605, -25.7098846, 35.6385460, -88.3343811, 92.9609299
3: -60.5642242, 77.7514114, -29.8839550, 41.5887413, -102.1529694, 107.6353683
4: -55.4830666, 78.0209122, -28.6537971, 40.7465019, -96.2295685, 106.6747131

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5936598, upper bound: 96.5160732
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952972, upper bound: 96.5178749
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.6879997, 71.5514069, -41.6879997, 71.5514069, -113.2394104, 113.2394028
1: -45.5929909, 60.5228043, -45.5929909, 60.5228043, -106.1157990, 106.1157990
2: -46.7689133, 60.5786438, -46.7689133, 60.5786438, -107.3475494, 107.3475571
3: -53.8531685, 70.0829163, -53.8531685, 70.0829163, -123.9360809, 123.9360809
4: -49.6292267, 70.1049042, -49.6292267, 70.1049042, -119.7341309, 119.7341309

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5482585
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544929, upper bound: 96.5544234
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.6879997, 71.5514069, -47.0005798, 78.9971008, -120.6851044, 118.5519867
1: -45.5929909, 60.5228043, -51.3436623, 67.0052643, -112.5982437, 111.8664703
2: -46.7689133, 60.5786438, -52.6201744, 67.1621475, -113.9310608, 113.1988068
3: -53.8531685, 70.0829163, -60.4751625, 77.6462936, -131.4994659, 130.5580750
4: -49.6292267, 70.1049042, -55.4036636, 77.9122620, -127.5414886, 125.5085678

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5728084
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5898227
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.0677299, 79.0901794, -41.6879997, 71.5514069, -118.6191406, 120.7781754
1: -51.4176712, 67.0940704, -45.5929909, 60.5228043, -111.9404755, 112.6870575
2: -52.6958504, 67.2510605, -46.7689133, 60.5786438, -113.2744827, 114.0199585
3: -60.5642242, 77.7514114, -53.8531685, 70.0829163, -130.6471405, 131.6045837
4: -55.4830666, 78.0209122, -49.6292267, 70.1049042, -125.5879669, 127.6501389

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5079238, upper bound: 96.5090491
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5944526, upper bound: 96.5658412
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.0677299, 79.0901794, -47.0677299, 79.0901794, -126.1579056, 126.1578827
1: -51.4176712, 67.0940704, -51.4176712, 67.0940704, -118.5117416, 118.5117416
2: -52.6958504, 67.2510605, -52.6958504, 67.2510605, -119.9468994, 119.9468994
3: -60.5642242, 77.7514114, -60.5642242, 77.7514114, -138.3156433, 138.3156433
4: -55.4830666, 78.0209122, -55.4830666, 78.0209122, -133.5039825, 133.5039825

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5079238, upper bound: 96.5885899
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5944526, upper bound: 96.5658412
time: 1.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.70 seconds
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5344041, upper bound: 96.4928805
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5474053, upper bound: 96.4953539
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5427896, upper bound: 96.4641733
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5433651, upper bound: 96.4663230
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5936598, upper bound: 96.5160732
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5952972, upper bound: 96.5178749
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5482585
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5544929, upper bound: 96.5544234
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5728084
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5898227
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5079238, upper bound: 96.5090491
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5944526, upper bound: 96.5658412
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5079238, upper bound: 96.5885899
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 4, lower bound: -96.5944526, upper bound: 96.5658412

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -22.1488571, 42.7812309, -84.1789017, 93.3872681
1: -45.2824707, 60.2273254, -24.2961674, 35.2251091, -80.5075836, 84.5234680
2: -46.4568481, 60.2715683, -24.9756222, 34.9029617, -81.3598099, 85.2471848
3: -53.5106544, 69.7326126, -29.0839233, 40.7468796, -94.2575378, 98.8165359
4: -49.3290863, 69.7293549, -27.9935036, 39.8805008, -89.2095871, 97.7228546

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4718376
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4928805
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -22.1488571, 42.7812309, -84.3456879, 93.6688995
1: -45.4678535, 60.4032860, -24.2961674, 35.2251091, -80.6929626, 84.6994400
2: -46.6354332, 60.4574471, -24.9756222, 34.9029617, -81.5383911, 85.4330597
3: -53.7209549, 69.9424973, -29.0839233, 40.7468796, -94.4678345, 99.0264206
4: -49.5006561, 69.9765320, -27.9935036, 39.8805008, -89.3811493, 97.9700241

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4743217
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4953539
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -43.8804359, 74.7824707, -18.7637291, 38.0417938, -81.9222260, 93.5461960
1: -47.9446640, 63.0959930, -20.6679897, 31.1780128, -79.1226730, 83.7639618
2: -49.1516342, 63.2080269, -21.2358284, 30.8240910, -79.9757004, 84.4438553
3: -56.5408173, 73.0602112, -24.9217834, 36.0716019, -92.6124191, 97.9819870
4: -51.8842621, 73.1964417, -24.4355717, 35.1456490, -87.0299072, 97.6320114

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4622789
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4641733
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -46.4467468, 77.4104614, -18.5472794, 37.5449486, -83.9916840, 95.9577332
1: -50.7226067, 65.8836670, -20.4270573, 30.8004913, -81.5231018, 86.3107224
2: -51.9846497, 66.0745773, -20.9914055, 30.4540539, -82.4387054, 87.0659790
3: -59.7645836, 76.3719864, -24.6313877, 35.6402893, -95.4048691, 101.0033722
4: -54.6952705, 76.7322388, -24.1622581, 34.7200470, -89.4152908, 100.8945007

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5375835, upper bound: 96.4644315
time: 2.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5375835, upper bound: 96.4663230
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.7764435, 75.9336700, -22.8273449, 43.5579681, -88.3343964, 98.7610168
1: -48.9274368, 64.2144089, -25.0134792, 35.9514008, -84.8788376, 89.2278900
2: -50.1477127, 64.3425446, -25.7098846, 35.6385460, -85.7862396, 90.0524292
3: -57.6820221, 74.3809357, -29.8839550, 41.5887413, -99.2707520, 104.2648926
4: -52.9067955, 74.5601578, -28.6537971, 40.7465019, -93.6532898, 103.2139511

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5936598, upper bound: 96.5160732
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5936598, upper bound: 96.5160732
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.3543434, 78.5891724, -22.5875759, 43.0374489, -90.3917923, 101.1767349
1: -51.7220154, 67.0382690, -24.7472591, 35.5508804, -87.2728882, 91.7855301
2: -52.9980049, 67.2279663, -25.4398346, 35.2447281, -88.2427292, 92.6678009
3: -60.9256325, 77.7400436, -29.5660572, 41.1308365, -102.0564728, 107.3060913
4: -55.7538834, 78.1223679, -28.3580475, 40.2921944, -96.0460663, 106.4804153

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952972, upper bound: 96.5178749
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952972, upper bound: 96.5178749
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -41.6879997, 71.5514069, -112.9490814, 112.9264069
1: -45.2824707, 60.2273254, -45.5929909, 60.5228043, -105.8052750, 105.8202972
2: -46.4568481, 60.2715683, -46.7689133, 60.5786438, -107.0354767, 107.0404816
3: -53.5106544, 69.7326126, -53.8531685, 70.0829163, -123.5935669, 123.5857849
4: -49.3290863, 69.7293549, -49.6292267, 70.1049042, -119.4339905, 119.3585815

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5306670, upper bound: 96.5403220
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5403220, upper bound: 96.5403220
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -41.6879997, 71.5514069, -113.1158676, 113.2080383
1: -45.4678535, 60.4032860, -45.5929909, 60.5228043, -105.9906616, 105.9962463
2: -46.6354332, 60.4574471, -46.7689133, 60.5786438, -107.2140656, 107.2263412
3: -53.7209549, 69.9424973, -53.8531685, 70.0829163, -123.8038712, 123.7956543
4: -49.5006561, 69.9765320, -49.6292267, 70.1049042, -119.6055527, 119.6057587

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5424076
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5544229
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -47.0005798, 78.9971008, -120.3947601, 118.2389832
1: -45.2824707, 60.2273254, -51.3436623, 67.0052643, -112.2877274, 111.5709763
2: -46.4568481, 60.2715683, -52.6201744, 67.1621475, -113.6189957, 112.8917389
3: -53.5106544, 69.7326126, -60.4751625, 77.6462936, -131.1569519, 130.2077789
4: -49.3290863, 69.7293549, -55.4036636, 77.9122620, -127.2413483, 125.1330185

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5502948, upper bound: 96.5713828
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5502948, upper bound: 96.5713828
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -47.0005798, 78.9971008, -120.5615540, 118.5206146
1: -45.4678535, 60.4032860, -51.3436623, 67.0052643, -112.4731140, 111.7469406
2: -46.6354332, 60.4574471, -52.6201744, 67.1621475, -113.7975769, 113.0776215
3: -53.7209549, 69.9424973, -60.4751625, 77.6462936, -131.3672485, 130.4176483
4: -49.5006561, 69.9765320, -55.4036636, 77.9122620, -127.4129105, 125.3801956

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5639249, upper bound: 96.5856623
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5639249, upper bound: 96.5898228
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -45.9752731, 78.6173096, -41.6879997, 71.5514069, -117.5266724, 120.3053055
1: -50.2814026, 66.3871689, -45.5929909, 60.5228043, -110.8042068, 111.9801559
2: -51.5448151, 66.4945831, -46.7689133, 60.5786438, -112.1234436, 113.2634964
3: -59.4520378, 76.8625641, -53.8531685, 70.0829163, -129.5349579, 130.7157288
4: -54.4311790, 77.0660400, -49.6292267, 70.1049042, -124.5360870, 126.6952667

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5707156, upper bound: 96.5545919
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5923098, upper bound: 96.5533050
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5932732, upper bound: 96.5650799
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.7436600, 76.0611954, -47.0677299, 79.0901794, -123.8338394, 123.1289215
1: -48.8998871, 64.2461166, -51.4176712, 67.0940704, -115.9939575, 115.6637878
2: -50.1368523, 64.3910141, -52.6958504, 67.2510605, -117.3878937, 117.0868530
3: -57.7000580, 74.4001617, -60.5642242, 77.7514114, -135.4514618, 134.9643860
4: -52.8640900, 74.5853195, -55.4830666, 78.0209122, -130.8849945, 130.0683746

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5586891, upper bound: 96.5668657
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5588345, upper bound: 96.5766460
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.9752731, 78.6173096, -47.0677299, 79.0901794, -125.0654526, 125.6850128
1: -50.2814026, 66.3871689, -51.4176712, 67.0940704, -117.3754654, 117.8048325
2: -51.5448151, 66.4945831, -52.6958504, 67.2510605, -118.7958450, 119.1904297
3: -59.4520378, 76.8625641, -60.5642242, 77.7514114, -137.2034454, 137.4267883
4: -54.4311790, 77.0660400, -55.4830666, 78.0209122, -132.4520874, 132.5491028

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6043788, upper bound: 96.5918336
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6101106, upper bound: 96.6108250
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.75 seconds
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4718376
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4928805
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4743217
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4953539
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4622789
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4641733
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5375835, upper bound: 96.4644315
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5375835, upper bound: 96.4663230
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5936598, upper bound: 96.5160732
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5936598, upper bound: 96.5160732
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5952972, upper bound: 96.5178749
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5952972, upper bound: 96.5178749
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5306670, upper bound: 96.5403220
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5403220, upper bound: 96.5403220
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5424076
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5544229
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5502948, upper bound: 96.5713828
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5502948, upper bound: 96.5713828
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5639249, upper bound: 96.5856623
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5639249, upper bound: 96.5898228
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5923098, upper bound: 96.5533050
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5932732, upper bound: 96.5650799
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5586891, upper bound: 96.5668657
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.5588345, upper bound: 96.5766460
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.6043788, upper bound: 96.5918336
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 4, lower bound: -96.6101106, upper bound: 96.6108250

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -18.3205643, 36.8871384, -78.2848053, 89.5589676
1: -45.2824707, 60.2273254, -20.1822853, 30.2803841, -75.5628510, 80.4095993
2: -46.4568481, 60.2715683, -20.7637596, 29.9831238, -76.4399643, 81.0353241
3: -53.5106544, 69.7326126, -24.3498306, 35.0611115, -88.5717621, 94.0824432
4: -49.3290863, 69.7293549, -23.8476677, 34.1320457, -83.4611359, 93.5770264

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4718376
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4850988, upper bound: 96.4644046
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -22.1621494, 42.7629700, -84.1606293, 93.4005585
1: -45.2824707, 60.2273254, -24.3082314, 35.2175751, -80.5000458, 84.5355301
2: -46.4568481, 60.2715683, -24.9882698, 34.9017639, -81.3586121, 85.2598190
3: -53.5106544, 69.7326126, -29.0971241, 40.7266769, -94.2373352, 98.8297348
4: -49.3290863, 69.7293549, -27.9738235, 39.8738594, -89.2029419, 97.7031784

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4918934
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4850988, upper bound: 96.4844611
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -18.3205643, 36.8871384, -78.4515991, 89.8405991
1: -45.4678535, 60.4032860, -20.1822853, 30.2803841, -75.7482300, 80.5855713
2: -46.6354332, 60.4574471, -20.7637596, 29.9831238, -76.6185532, 81.2212067
3: -53.7209549, 69.9424973, -24.3498306, 35.0611115, -88.7820511, 94.2923279
4: -49.5006561, 69.9765320, -23.8476677, 34.1320457, -83.6326904, 93.8241882

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4743217
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4943251, upper bound: 96.4653808
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -22.1621494, 42.7629700, -84.3274231, 93.6821899
1: -45.4678535, 60.4032860, -24.3082314, 35.2175751, -80.6854248, 84.7114944
2: -46.6354332, 60.4574471, -24.9882698, 34.9017639, -81.5372009, 85.4457016
3: -53.7209549, 69.9424973, -29.0971241, 40.7266769, -94.4476242, 99.0396194
4: -49.5006561, 69.9765320, -27.9738235, 39.8738594, -89.3745117, 97.9503555

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4943667
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4943251, upper bound: 96.4854372
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -43.8804359, 74.7824707, -16.9044724, 35.6302986, -79.5107346, 91.6869354
1: -47.9446640, 63.0959930, -18.6737976, 28.9525528, -76.8972015, 81.7697754
2: -49.1516342, 63.2080269, -19.1880455, 28.6019154, -77.7535477, 82.3960724
3: -56.5408173, 73.0602112, -22.6492405, 33.4476280, -89.9884491, 95.7094498
4: -51.8842621, 73.1964417, -22.4624233, 32.5191383, -84.4033966, 95.6588593

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5354677, upper bound: 96.4621473
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250448, upper bound: 96.4582251
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4622695
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -43.8804359, 74.7824707, -19.0102005, 37.3557739, -81.2362061, 93.7926636
1: -47.9446640, 63.0959930, -20.8693199, 30.9446335, -78.8892822, 83.9653091
2: -49.1516342, 63.2080269, -21.4940758, 30.6324863, -79.7841110, 84.7021027
3: -56.5408173, 73.0602112, -25.0909214, 35.8480148, -92.3888168, 98.1511230
4: -51.8842621, 73.1964417, -24.5277863, 34.9815598, -86.8658142, 97.7242279

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5354677, upper bound: 96.4640296
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250448, upper bound: 96.4604866
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4641597
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.4467468, 77.4104614, -16.9044724, 35.6302986, -82.0770340, 94.3149338
1: -50.7226067, 65.8836670, -18.6737976, 28.9525528, -79.6751556, 84.5574646
2: -51.9846497, 66.0745773, -19.1880455, 28.6019154, -80.5865631, 85.2626190
3: -59.7645836, 76.3719864, -22.6492405, 33.4476280, -93.2122116, 99.0212173
4: -54.6952705, 76.7322388, -22.4624233, 32.5191383, -87.2143936, 99.1946640

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5358331, upper bound: 96.4643292
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5354899, upper bound: 96.4614836
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5088741, upper bound: 96.4576761
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.4467468, 77.4104614, -19.0102005, 37.3557739, -83.8025208, 96.4206619
1: -50.7226067, 65.8836670, -20.8693199, 30.9446335, -81.6672363, 86.7529907
2: -51.9846497, 66.0745773, -21.4940758, 30.6324863, -82.6171341, 87.5686493
3: -59.7645836, 76.3719864, -25.0909214, 35.8480148, -95.6125946, 101.4628983
4: -54.6952705, 76.7322388, -24.5277863, 34.9815598, -89.6768112, 101.2600250

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5358331, upper bound: 96.4662150
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5354899, upper bound: 96.4614836
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5088741, upper bound: 96.4598620
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -44.7764435, 75.9336700, -20.8574600, 40.9532814, -85.7297211, 96.7911224
1: -48.9274368, 64.2144089, -22.9023209, 33.5480461, -82.4754791, 87.1167297
2: -50.1477127, 64.3425446, -23.5333061, 33.2290497, -83.3767624, 87.8758545
3: -57.6820221, 74.3809357, -27.4619255, 38.7649078, -96.4469223, 101.8428574
4: -52.9067955, 74.5601578, -26.5332832, 37.8997345, -90.8065109, 101.0934296

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5385161, upper bound: 96.4649537
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4892047
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5912601, upper bound: 96.5060198
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -44.7764435, 75.9336700, -23.1581726, 42.8691025, -87.6455460, 99.0918427
1: -48.9274368, 64.2144089, -25.3201675, 35.7193375, -84.6467743, 89.5345764
2: -50.1477127, 64.3425446, -26.0323143, 35.4587212, -85.6064224, 90.3748627
3: -57.6820221, 74.3809357, -30.1483173, 41.3781395, -99.0601654, 104.5292358
4: -52.9067955, 74.5601578, -28.7969799, 40.6037445, -93.5105286, 103.3571243

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5385161, upper bound: 96.4649537
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4892047
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5912601, upper bound: 96.5060198
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.3543434, 78.5891724, -20.8574600, 40.9532814, -88.3076248, 99.4466248
1: -51.7220154, 67.0382690, -22.9023209, 33.5480461, -85.2700577, 89.9405899
2: -52.9980049, 67.2279663, -23.5333061, 33.2290497, -86.2270508, 90.7612686
3: -60.9256325, 77.7400436, -27.4619255, 38.7649078, -99.6905365, 105.2019653
4: -55.7538834, 78.1223679, -26.5332832, 37.8997345, -93.6535950, 104.6556549

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5549355, upper bound: 96.5007648
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5927767, upper bound: 96.5077185
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.3543434, 78.5891724, -23.1581726, 42.8691025, -90.2234497, 101.7473297
1: -51.7220154, 67.0382690, -25.3201675, 35.7193375, -87.4413528, 92.3584290
2: -52.9980049, 67.2279663, -26.0323143, 35.4587212, -88.4567032, 93.2602768
3: -60.9256325, 77.7400436, -30.1483173, 41.3781395, -102.3037720, 107.8883438
4: -55.7538834, 78.1223679, -28.7969799, 40.6037445, -96.3576202, 106.9193497

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5549355, upper bound: 96.5007648
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5927767, upper bound: 96.5077185
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -41.3976822, 71.2384109, -112.6360779, 112.6360855
1: -45.2824707, 60.2273254, -45.2824707, 60.2273254, -105.5097733, 105.5097733
2: -46.4568481, 60.2715683, -46.4568481, 60.2715683, -106.7284012, 106.7284012
3: -53.5106544, 69.7326126, -53.5106544, 69.7326126, -123.2432709, 123.2432709
4: -49.3290863, 69.7293549, -49.3290863, 69.7293549, -119.0584412, 119.0584412

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369726, upper bound: 96.5385528
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5389451, upper bound: 96.5389452
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -41.5644608, 71.5200424, -112.9177094, 112.8028717
1: -45.2824707, 60.2273254, -45.4678535, 60.4032860, -105.6857376, 105.6951675
2: -46.4568481, 60.2715683, -46.6354332, 60.4574471, -106.9142761, 106.9069901
3: -53.5106544, 69.7326126, -53.7209549, 69.9424973, -123.4531555, 123.4535675
4: -49.3290863, 69.7293549, -49.5006561, 69.9765320, -119.3056183, 119.2300110

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369726, upper bound: 96.5472753
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5389451, upper bound: 96.5389452
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -41.3976822, 71.2384109, -112.8028717, 112.9177017
1: -45.4678535, 60.4032860, -45.2824707, 60.2273254, -105.6951599, 105.6857376
2: -46.6354332, 60.4574471, -46.4568481, 60.2715683, -106.9069901, 106.9142761
3: -53.7209549, 69.9424973, -53.5106544, 69.7326126, -123.4535599, 123.4531555
4: -49.5006561, 69.9765320, -49.3290863, 69.7293549, -119.2300110, 119.3056183

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5478017, upper bound: 96.5403354
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5515338, upper bound: 96.5410762
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -41.5644608, 71.5200424, -113.0845032, 113.0845032
1: -45.4678535, 60.4032860, -45.4678535, 60.4032860, -105.8711166, 105.8711166
2: -46.6354332, 60.4574471, -46.6354332, 60.4574471, -107.0928726, 107.0928726
3: -53.7209549, 69.9424973, -53.7209549, 69.9424973, -123.6634445, 123.6634369
4: -49.5006561, 69.9765320, -49.5006561, 69.9765320, -119.4771729, 119.4771881

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5478023, upper bound: 96.5528074
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5515343, upper bound: 96.5535928
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -46.6724052, 78.6447525, -120.0424042, 117.9108124
1: -45.2824707, 60.2273254, -50.9914055, 66.6645966, -111.9470673, 111.2187271
2: -46.4568481, 60.2715683, -52.2635460, 66.8147812, -113.2716293, 112.5351105
3: -53.5106544, 69.7326126, -60.0874786, 77.2385712, -130.7492218, 129.8200989
4: -49.3290863, 69.7293549, -55.0474586, 77.4880524, -126.8171387, 124.7768097

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5439511, upper bound: 96.5616740
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5615340
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.3976822, 71.2384109, -46.8382721, 78.9225159, -120.3201828, 118.0766754
1: -45.2824707, 60.2273254, -51.1744385, 66.8517227, -112.1341858, 111.4017563
2: -46.4568481, 60.2715683, -52.4467583, 66.9971619, -113.4540100, 112.7183075
3: -53.5106544, 69.7326126, -60.2932091, 77.4673004, -130.9779510, 130.0258179
4: -49.3290863, 69.7293549, -55.2447815, 77.7271042, -127.0561905, 124.9741364

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5439512, upper bound: 96.5616742
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5660651
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -46.6724052, 78.6447525, -120.2092133, 118.1924438
1: -45.4678535, 60.4032860, -50.9914055, 66.6645966, -112.1324463, 111.3946838
2: -46.6354332, 60.4574471, -52.2635460, 66.8147812, -113.4502106, 112.7209778
3: -53.7209549, 69.9424973, -60.0874786, 77.2385712, -130.9595184, 130.0299530
4: -49.5006561, 69.9765320, -55.0474586, 77.4880524, -126.9886932, 125.0239716

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5595087, upper bound: 96.5840747
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5631831, upper bound: 96.5849429
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -41.5644608, 71.5200424, -46.8382721, 78.9225159, -120.4869766, 118.3583069
1: -45.4678535, 60.4032860, -51.1744385, 66.8517227, -112.3195648, 111.5777206
2: -46.6354332, 60.4574471, -52.4467583, 66.9971619, -113.6325989, 112.9041901
3: -53.7209549, 69.9424973, -60.2932091, 77.4673004, -131.1882477, 130.2356873
4: -49.5006561, 69.9765320, -55.2447815, 77.7271042, -127.2277451, 125.2213135

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5595092, upper bound: 96.5840748
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5631836, upper bound: 96.5849430
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -45.9752731, 78.6173096, -41.3976822, 71.2384109, -117.2136765, 120.0149765
1: -50.2814026, 66.3871689, -45.2824707, 60.2273254, -110.5087051, 111.6696396
2: -51.5448151, 66.4945831, -46.4568481, 60.2715683, -111.8163681, 112.9514313
3: -59.4520378, 76.8625641, -53.5106544, 69.7326126, -129.1846466, 130.3731995
4: -54.4311790, 77.0660400, -49.3290863, 69.7293549, -124.1605377, 126.3951263

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5883000, upper bound: 96.5528812
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5883000, upper bound: 96.5528812
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -45.9752731, 78.6173096, -41.5644608, 71.5200424, -117.4953079, 120.1817703
1: -50.2814026, 66.3871689, -45.4678535, 60.4032860, -110.6846619, 111.8550186
2: -51.5448151, 66.4945831, -46.6354332, 60.4574471, -112.0022354, 113.1300201
3: -59.4520378, 76.8625641, -53.7209549, 69.9424973, -129.3945312, 130.5835114
4: -54.4311790, 77.0660400, -49.5006561, 69.9765320, -124.4077148, 126.5666962

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5893241, upper bound: 96.5646140
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5893241, upper bound: 96.5650805
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -44.7436600, 76.0611954, -45.9628410, 77.8067627, -122.5504150, 122.0240326
1: -48.8998871, 64.2461166, -50.2656403, 66.0334091, -114.9332962, 114.5117569
2: -50.1368523, 64.3910141, -51.4855194, 66.1813202, -116.3181686, 115.8765335
3: -57.7000580, 74.4001617, -59.2746925, 76.5063095, -134.2063446, 133.6748505
4: -52.8640900, 74.5853195, -54.2514763, 76.7217941, -129.5858612, 128.8367920

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5586891, upper bound: 96.5668663
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5586891, upper bound: 96.5668663
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -44.7436600, 76.0611954, -45.3854561, 76.8448334, -121.5884933, 121.4466400
1: -48.8998871, 64.2461166, -49.6261864, 65.1088181, -114.0086975, 113.8722992
2: -50.1368523, 64.3910141, -50.8572235, 65.2283630, -115.3651886, 115.2482376
3: -57.7000580, 74.4001617, -58.5323372, 75.4720154, -133.1720428, 132.9324951
4: -52.8640900, 74.5853195, -53.6500931, 75.6274261, -128.4914856, 128.2354126

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5588345, upper bound: 96.5766460
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5588345, upper bound: 96.5766460
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -45.9752731, 78.6173096, -45.9628410, 77.8067627, -123.7820282, 124.5801392
1: -50.2814026, 66.3871689, -50.2656403, 66.0334091, -116.3148117, 116.6528091
2: -51.5448151, 66.4945831, -51.4855194, 66.1813202, -117.7261200, 117.9801025
3: -59.4520378, 76.8625641, -59.2746925, 76.5063095, -135.9583435, 136.1372528
4: -54.4311790, 77.0660400, -54.2514763, 76.7217941, -131.1529694, 131.3174896

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5895394, upper bound: 96.5895394
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5895394, upper bound: 96.5918336
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -45.9752731, 78.6173096, -45.3854561, 76.8448334, -122.8201065, 124.0027313
1: -50.2814026, 66.3871689, -49.6261864, 65.1088181, -115.3902130, 116.0133514
2: -51.5448151, 66.4945831, -50.8572235, 65.2283630, -116.7731552, 117.3518066
3: -59.4520378, 76.8625641, -58.5323372, 75.4720154, -134.9240417, 135.3948975
4: -54.4311790, 77.0660400, -53.6500931, 75.6274261, -130.0586090, 130.7161255

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5895498, upper bound: 96.6022795
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5895499, upper bound: 96.6108251
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.58 seconds
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4718376
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.4850988, upper bound: 96.4644046
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5275494, upper bound: 96.4918934
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.4850988, upper bound: 96.4844611
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4743217
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.4943251, upper bound: 96.4653808
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5405957, upper bound: 96.4943667
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.4943251, upper bound: 96.4854372
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5250448, upper bound: 96.4582251
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4622695
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5250448, upper bound: 96.4604866
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5369123, upper bound: 96.4641597
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5354899, upper bound: 96.4614836
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5088741, upper bound: 96.4576761
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5354899, upper bound: 96.4614836
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5088741, upper bound: 96.4598620
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4892047
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5912601, upper bound: 96.5060198
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4892047
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5912601, upper bound: 96.5060198
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5549355, upper bound: 96.5007648
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5927767, upper bound: 96.5077185
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5549355, upper bound: 96.5007648
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5927767, upper bound: 96.5077185
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5369726, upper bound: 96.5385528
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5389451, upper bound: 96.5389452
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5369726, upper bound: 96.5472753
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5389451, upper bound: 96.5389452
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5478017, upper bound: 96.5403354
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5515338, upper bound: 96.5410762
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5478023, upper bound: 96.5528074
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5515343, upper bound: 96.5535928
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5439511, upper bound: 96.5616740
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5615340
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5439512, upper bound: 96.5616742
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5660651
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5595087, upper bound: 96.5840747
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5631831, upper bound: 96.5849429
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5595092, upper bound: 96.5840748
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5631836, upper bound: 96.5849430
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5883000, upper bound: 96.5528812
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5883000, upper bound: 96.5528812
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5893241, upper bound: 96.5646140
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5893241, upper bound: 96.5650805
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5586891, upper bound: 96.5668663
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5586891, upper bound: 96.5668663
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5588345, upper bound: 96.5766460
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5588345, upper bound: 96.5766460
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5895394, upper bound: 96.5895394
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5895394, upper bound: 96.5918336
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5895498, upper bound: 96.6022795
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.58
Output dim: 4, lower bound: -96.5895499, upper bound: 96.6108251

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.1704483, 71.1371536, -18.2830048, 36.8303566, -78.0008087, 89.4201584
1: -45.0442657, 59.9962807, -20.1415844, 30.2278023, -75.2720642, 80.1378632
2: -46.2192459, 60.0251732, -20.7216034, 29.9309063, -76.1501541, 80.7467651
3: -53.2860374, 69.4535065, -24.3027287, 34.9990501, -88.2850876, 93.7562332
4: -49.0888138, 69.4806747, -23.8026047, 34.0721016, -83.1608887, 93.2832794

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5269075, upper bound: 96.4664982
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5269075, upper bound: 96.4664982
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.1704483, 71.1371536, -22.1255817, 42.7070541, -83.8775024, 93.2627335
1: -45.0442657, 59.9962807, -24.2683620, 35.1657562, -80.2100143, 84.2646408
2: -46.2192459, 60.0251732, -24.9472008, 34.8503914, -81.0696411, 84.9723740
3: -53.2860374, 69.4535065, -29.0510216, 40.6655502, -93.9515839, 98.5045319
4: -49.0888138, 69.4806747, -27.9293518, 39.8148613, -88.9036560, 97.4100266

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5343734, upper bound: 96.4918933
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5343734, upper bound: 96.4918933
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.3172379, 71.3980560, -18.2830048, 36.8303566, -78.1475983, 89.6810455
1: -45.2076569, 60.1511192, -20.1415844, 30.2278023, -75.4354553, 80.2926941
2: -46.3758087, 60.1892281, -20.7216034, 29.9309063, -76.3067169, 80.9108276
3: -53.4712257, 69.6379395, -24.3027287, 34.9990501, -88.4702759, 93.9406662
4: -49.2401886, 69.7003784, -23.8026047, 34.0721016, -83.3122787, 93.5029755

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5399537, upper bound: 96.4689821
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5399543, upper bound: 96.4743218
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.3172379, 71.3980560, -22.1255817, 42.7070541, -84.0242767, 93.5236282
1: -45.2076569, 60.1511192, -24.2683620, 35.1657562, -80.3734055, 84.4194794
2: -46.3758087, 60.1892281, -24.9472008, 34.8503914, -81.2261963, 85.1364288
3: -53.4712257, 69.6379395, -29.0510216, 40.6655502, -94.1367722, 98.6889572
4: -49.2401886, 69.7003784, -27.9293518, 39.8148613, -89.0550537, 97.6297302

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5473771, upper bound: 96.4943662
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5473777, upper bound: 96.4943663
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -38.3263359, 65.9709396, -15.9135332, 34.3209076, -72.6472244, 81.8844528
1: -41.8615341, 55.3545837, -17.5958881, 27.7787437, -69.6402664, 72.9504547
2: -42.8908043, 55.3991508, -18.1067772, 27.4292793, -70.3200836, 73.5059280
3: -49.3152542, 64.0527115, -21.4440689, 32.0616341, -81.3768921, 85.4967804
4: -45.4549065, 64.0283585, -21.3869438, 31.1493702, -76.6042786, 85.4152985

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5233642, upper bound: 96.4556906
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5231265, upper bound: 96.4546136
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -41.9352303, 72.2366638, -16.8822155, 35.5993881, -77.5346069, 89.1188660
1: -45.8540802, 60.7566032, -18.6505699, 28.9249229, -74.7790070, 79.4071732
2: -47.0273743, 60.8546638, -19.1636715, 28.5741501, -75.6015167, 80.0183334
3: -54.1787529, 70.3179169, -22.6224461, 33.4152527, -87.5940018, 92.9403534
4: -49.6724319, 70.4120941, -22.4373722, 32.4869461, -82.1593781, 92.8494644

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5293271, upper bound: 96.4577992
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5265099, upper bound: 96.4559807
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -38.3263359, 65.9709396, -17.6556206, 35.6560173, -73.9823303, 83.6265564
1: -41.8615341, 55.3545837, -19.4301872, 29.3884926, -71.2500153, 74.7847443
2: -42.8908043, 55.3991508, -20.0265064, 29.0668945, -71.9576874, 75.4256516
3: -49.3152542, 64.0527115, -23.4758434, 34.0137367, -83.3289948, 87.5285416
4: -45.4549065, 64.0283585, -23.0961151, 33.1469498, -78.6018524, 87.1244736

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5302431, upper bound: 96.4585090
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5297693, upper bound: 96.4567572
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -41.9352303, 72.2366638, -18.9893417, 37.3273239, -79.2625351, 91.2259827
1: -45.8540802, 60.7566032, -20.8469772, 30.9198456, -76.7739258, 81.6035767
2: -47.0273743, 60.8546638, -21.4711533, 30.6078358, -77.6352081, 82.3258209
3: -54.1787529, 70.3179169, -25.0661983, 35.8190079, -89.9977417, 95.3841095
4: -49.6724319, 70.4120941, -24.5044289, 34.9526138, -84.6250458, 94.9165039

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5368971, upper bound: 96.4604002
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5352137, upper bound: 96.4580604
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5271949, upper bound: 96.4350133
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5426262, upper bound: 96.4641597
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -45.0994148, 75.1094742, -16.9044724, 35.6302986, -80.7297134, 92.0139389
1: -49.2443237, 63.9585648, -18.6737976, 28.9525528, -78.1968536, 82.6323624
2: -50.4704399, 64.1412659, -19.1880455, 28.6019154, -79.0723572, 83.3293152
3: -58.0179062, 74.1450882, -22.6492405, 33.4476280, -91.4655304, 96.7943268
4: -53.1352463, 74.4839401, -22.4624233, 32.5191383, -85.6543808, 96.9463501

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5063253, upper bound: 96.4517727
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5063253, upper bound: 96.4576761
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -45.0994148, 75.1094742, -19.0102005, 37.3557739, -82.4551849, 94.1196671
1: -49.2443237, 63.9585648, -20.8693199, 30.9446335, -80.1889496, 84.8278809
2: -50.4704399, 64.1412659, -21.4940758, 30.6324863, -81.1029282, 85.6353455
3: -58.0179062, 74.1450882, -25.0909214, 35.8480148, -93.8659134, 99.2360077
4: -53.1352463, 74.4839401, -24.5277863, 34.9815598, -88.1167984, 99.0117188

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5041604, upper bound: 96.4404820
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5041604, upper bound: 96.4598620
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.7973251, 75.6151886, -20.8574600, 40.9532814, -84.7506027, 96.4726486
1: -47.9171906, 63.6562996, -22.9023209, 33.5480461, -81.4652405, 86.5585938
2: -49.1294861, 63.7237816, -23.5333061, 33.2290497, -82.3585281, 87.2570877
3: -56.7174110, 73.6727219, -27.4619255, 38.7649078, -95.4823151, 101.1346436
4: -52.0038490, 73.7733383, -26.5332832, 37.8997345, -89.9035797, 100.3066177

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5904166, upper bound: 96.4931919
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5894997, upper bound: 96.5023869
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.7973251, 75.6151886, -23.1581726, 42.8691025, -86.6664200, 98.7733612
1: -47.9171906, 63.6562996, -25.3201675, 35.7193375, -83.6365280, 88.9764404
2: -49.1294861, 63.7237816, -26.0323143, 35.4587212, -84.5881958, 89.7560959
3: -56.7174110, 73.6727219, -30.1483173, 41.3781395, -98.0955505, 103.8210297
4: -52.0038490, 73.7733383, -28.7969799, 40.6037445, -92.6075897, 102.5703125

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5895980, upper bound: 96.5039087
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5905815, upper bound: 96.5019544
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5907385, upper bound: 96.5039227
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5907385, upper bound: 96.5060198
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -45.2390633, 75.7938309, -20.8574600, 40.9532814, -86.1923447, 96.6512909
1: -49.4263840, 64.4019470, -22.9023209, 33.5480461, -82.9744263, 87.3042603
2: -50.6609344, 64.6012115, -23.5333061, 33.2290497, -83.8899841, 88.1345139
3: -58.3063240, 74.6300278, -27.4619255, 38.7649078, -97.0712280, 102.0919495
4: -53.3483582, 74.9629974, -26.5332832, 37.8997345, -91.2480927, 101.4962769

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5512023, upper bound: 96.4971652
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5167449, upper bound: 96.4828151
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -46.1301308, 77.8321686, -20.8574600, 40.9532814, -87.0834122, 98.6896286
1: -50.4351044, 66.0740509, -22.9023209, 33.5480461, -83.9831238, 88.9763641
2: -51.6798859, 66.2257919, -23.5333061, 33.2290497, -84.9089355, 89.7590942
3: -59.6054382, 76.5463333, -27.4619255, 38.7649078, -98.3703461, 104.0082550
4: -54.5248718, 76.8399048, -26.5332832, 37.8997345, -92.4246063, 103.3731842

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5923059, upper bound: 96.5049281
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5877020, upper bound: 96.4983632
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -45.2390633, 75.7938309, -23.1581726, 42.8691025, -88.1081619, 98.9519958
1: -49.4263840, 64.4019470, -25.3201675, 35.7193375, -85.1457214, 89.7221069
2: -50.6609344, 64.6012115, -26.0323143, 35.4587212, -86.1196442, 90.6335220
3: -58.3063240, 74.6300278, -30.1483173, 41.3781395, -99.6844635, 104.7783356
4: -53.3483582, 74.9629974, -28.7969799, 40.6037445, -93.9521027, 103.7599716

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5513752, upper bound: 96.4953571
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5550602, upper bound: 96.4988350
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5550602, upper bound: 96.5007648
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -46.1301308, 77.8321686, -23.1581726, 42.8691025, -88.9992294, 100.9903412
1: -50.4351044, 66.0740509, -25.3201675, 35.7193375, -86.1544418, 91.3942108
2: -51.6798859, 66.2257919, -26.0323143, 35.4587212, -87.1385880, 92.2581024
3: -59.6054382, 76.5463333, -30.1483173, 41.3781395, -100.9835815, 106.6946411
4: -54.5248718, 76.8399048, -28.7969799, 40.6037445, -95.1286163, 105.6368790

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5913249, upper bound: 96.5026089
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5912002, upper bound: 96.5050850
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5920981, upper bound: 96.5036420
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5888331, upper bound: 96.5043087
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5896020, upper bound: 96.5075721
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -41.3976822, 71.2384109, -112.3924484, 112.2605209
1: -45.0143814, 59.8876724, -45.2824707, 60.2273254, -105.2416840, 105.1701431
2: -46.1856041, 59.9343796, -46.4568481, 60.2715683, -106.4571686, 106.3912277
3: -53.1999969, 69.3314743, -53.5106544, 69.7326126, -122.9326096, 122.8421326
4: -49.0350761, 69.3348389, -49.3290863, 69.7293549, -118.7644348, 118.6639252

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5365802, upper bound: 96.5365802
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5365802, upper bound: 96.5365802
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -41.3976822, 71.2384109, -112.3510056, 111.4631195
1: -44.9368896, 59.3829651, -45.2824707, 60.2273254, -105.1641998, 104.6654358
2: -46.0977135, 59.4603043, -46.4568481, 60.2715683, -106.3692627, 105.9171524
3: -53.0495338, 68.7801971, -53.5106544, 69.7326126, -122.7821503, 122.2908478
4: -48.8637085, 68.8170929, -49.3290863, 69.7293549, -118.5930634, 118.1461639

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5252640, upper bound: 96.4903224
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5346792, upper bound: 96.5346792
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -41.5644608, 71.5200424, -112.6740799, 112.4273071
1: -45.0143814, 59.8876724, -45.4678535, 60.4032860, -105.4176559, 105.3555298
2: -46.1856041, 59.9343796, -46.6354332, 60.4574471, -106.6430359, 106.5698090
3: -53.1999969, 69.3314743, -53.7209549, 69.9424973, -123.1424942, 123.0524139
4: -49.0350761, 69.3348389, -49.5006561, 69.9765320, -119.0115967, 118.8354797

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5376081, upper bound: 96.5453321
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5376081, upper bound: 96.5453316
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -41.5644608, 71.5200424, -112.6326447, 111.6299057
1: -44.9368896, 59.3829651, -45.4678535, 60.4032860, -105.3401642, 104.8508148
2: -46.0977135, 59.4603043, -46.6354332, 60.4574471, -106.5551376, 106.0957336
3: -53.0495338, 68.7801971, -53.7209549, 69.9424973, -122.9920349, 122.5011444
4: -48.8637085, 68.8170929, -49.5006561, 69.9765320, -118.8402405, 118.3177261

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5395613, upper bound: 96.5456024
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5395613, upper bound: 96.5475340
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -41.3976822, 71.2384109, -112.5592651, 112.5422897
1: -45.1999130, 60.0637360, -45.2824707, 60.2273254, -105.4272308, 105.3461838
2: -46.3642349, 60.1203766, -46.4568481, 60.2715683, -106.6358032, 106.5772247
3: -53.4104080, 69.5413132, -53.5106544, 69.7326126, -123.1430206, 123.0519714
4: -49.2065659, 69.5821609, -49.3290863, 69.7293549, -118.9359207, 118.9112396

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5474098, upper bound: 96.5383628
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5474098, upper bound: 96.5403354
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -41.3976822, 71.2384109, -112.5156174, 111.7386093
1: -45.1196556, 59.5536995, -45.2824707, 60.2273254, -105.3469543, 104.8361664
2: -46.2735367, 59.6405334, -46.4568481, 60.2715683, -106.5451050, 106.0973740
3: -53.2558594, 68.9840393, -53.5106544, 69.7326126, -122.9884720, 122.4946899
4: -49.0312271, 69.0583344, -49.3290863, 69.7293549, -118.7605820, 118.3874207

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5511419, upper bound: 96.5391036
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5511419, upper bound: 96.5410762
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -41.5644608, 71.5200424, -112.8409119, 112.7090912
1: -45.1999130, 60.0637360, -45.4678535, 60.4032860, -105.6031876, 105.5315704
2: -46.3642349, 60.1203766, -46.6354332, 60.4574471, -106.8216629, 106.7558136
3: -53.4104080, 69.5413132, -53.7209549, 69.9424973, -123.3529053, 123.2622528
4: -49.2065659, 69.5821609, -49.5006561, 69.9765320, -119.1830978, 119.0828094

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5491919, upper bound: 96.5491859
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5491919, upper bound: 96.5528045
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -41.5644608, 71.5200424, -112.7972412, 111.9054031
1: -45.1196556, 59.5536995, -45.4678535, 60.4032860, -105.5229111, 105.0215530
2: -46.2735367, 59.6405334, -46.6354332, 60.4574471, -106.7309799, 106.2759705
3: -53.2558594, 68.9840393, -53.7209549, 69.9424973, -123.1983566, 122.7049866
4: -49.0312271, 69.0583344, -49.5006561, 69.9765320, -119.0077438, 118.5589752

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5529240, upper bound: 96.5499252
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5529240, upper bound: 96.5535922
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -46.6724052, 78.6447525, -119.7987976, 117.5352325
1: -45.0143814, 59.8876724, -50.9914055, 66.6645966, -111.6789703, 110.8790741
2: -46.1856041, 59.9343796, -52.2635460, 66.8147812, -113.0003815, 112.1979218
3: -53.1999969, 69.3314743, -60.0874786, 77.2385712, -130.4385529, 129.4189453
4: -49.0350761, 69.3348389, -55.0474586, 77.4880524, -126.5231171, 124.3822784

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5439511, upper bound: 96.5615244
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5439511, upper bound: 96.5615246
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -46.6724052, 78.6447525, -119.7573547, 116.7378540
1: -44.9368896, 59.3829651, -50.9914055, 66.6645966, -111.6014862, 110.3743744
2: -46.0977135, 59.4603043, -52.2635460, 66.8147812, -112.9124908, 111.7238464
3: -53.0495338, 68.7801971, -60.0874786, 77.2385712, -130.2881012, 128.8676605
4: -48.8637085, 68.8170929, -55.0474586, 77.4880524, -126.3517609, 123.8645325

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5615339
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5615341
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -46.8382721, 78.9225159, -120.0765610, 117.7011108
1: -45.0143814, 59.8876724, -51.1744385, 66.8517227, -111.8661041, 111.0621109
2: -46.1856041, 59.9343796, -52.4467583, 66.9971619, -113.1827698, 112.3811340
3: -53.1999969, 69.3314743, -60.2932091, 77.4673004, -130.6672974, 129.6246796
4: -49.0350761, 69.3348389, -55.2447815, 77.7271042, -126.7621689, 124.5796204

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5447307, upper bound: 96.5660459
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5447307, upper bound: 96.5660459
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -46.8382721, 78.9225159, -120.0351181, 116.9037170
1: -44.9368896, 59.3829651, -51.1744385, 66.8517227, -111.7886124, 110.5574036
2: -46.0977135, 59.4603043, -52.4467583, 66.9971619, -113.0948792, 111.9070587
3: -53.0495338, 68.7801971, -60.2932091, 77.4673004, -130.5168304, 129.0734100
4: -48.8637085, 68.8170929, -55.2447815, 77.7271042, -126.5908127, 124.0618744

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5465698, upper bound: 96.5660649
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5465698, upper bound: 96.5660649
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -46.6724052, 78.6447525, -119.9656143, 117.8170319
1: -45.1999130, 60.0637360, -50.9914055, 66.6645966, -111.8645096, 111.0551300
2: -46.3642349, 60.1203766, -52.2635460, 66.8147812, -113.1790161, 112.3839264
3: -53.4104080, 69.5413132, -60.0874786, 77.2385712, -130.6489716, 129.6287842
4: -49.2065659, 69.5821609, -55.0474586, 77.4880524, -126.6946182, 124.6296234

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5594923, upper bound: 96.5832365
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5594923, upper bound: 96.5840747
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -46.6724052, 78.6447525, -119.9219513, 117.0133514
1: -45.1196556, 59.5536995, -50.9914055, 66.6645966, -111.7842407, 110.5451050
2: -46.2735367, 59.6405334, -52.2635460, 66.8147812, -113.0883179, 111.9040833
3: -53.2558594, 68.9840393, -60.0874786, 77.2385712, -130.4944153, 129.0715179
4: -49.0312271, 69.0583344, -55.0474586, 77.4880524, -126.5192566, 124.1057739

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5631717, upper bound: 96.5840595
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5631717, upper bound: 96.5849430
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -46.8382721, 78.9225159, -120.2433777, 117.9828949
1: -45.1999130, 60.0637360, -51.1744385, 66.8517227, -112.0516357, 111.2381668
2: -46.3642349, 60.1203766, -52.4467583, 66.9971619, -113.3613968, 112.5671387
3: -53.4104080, 69.5413132, -60.2932091, 77.4673004, -130.8777008, 129.8345184
4: -49.2065659, 69.5821609, -55.2447815, 77.7271042, -126.9336700, 124.8269424

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5598442, upper bound: 96.5870418
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5598442, upper bound: 96.5880394
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -46.8382721, 78.9225159, -120.1997147, 117.1792145
1: -45.1196556, 59.5536995, -51.1744385, 66.8517227, -111.9713593, 110.7281342
2: -46.2735367, 59.6405334, -52.4467583, 66.9971619, -113.2706985, 112.0872803
3: -53.2558594, 68.9840393, -60.2932091, 77.4673004, -130.7231598, 129.2772369
4: -49.0312271, 69.0583344, -55.2447815, 77.7271042, -126.7583237, 124.3031158

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5635216, upper bound: 96.5879015
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5635216, upper bound: 96.5889178
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -45.6615295, 78.2715302, -41.3976822, 71.2384109, -116.8999252, 119.6692123
1: -49.9450302, 66.0585632, -45.2824707, 60.2273254, -110.1723251, 111.3410187
2: -51.2054443, 66.1545029, -46.4568481, 60.2715683, -111.4770050, 112.6113510
3: -59.0777664, 76.4735184, -53.5106544, 69.7326126, -128.8103790, 129.9841614
4: -54.0997353, 76.6523056, -49.3290863, 69.7293549, -123.8290863, 125.9813843

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5496724
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5875136, upper bound: 96.5516410
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -45.7164001, 78.4372253, -41.3976822, 71.2384109, -116.9547882, 119.8348923
1: -50.0085220, 66.1290207, -45.2824707, 60.2273254, -110.2358398, 111.4114914
2: -51.2662811, 66.2252960, -46.4568481, 60.2715683, -111.5378342, 112.6821442
3: -59.1561966, 76.5591431, -53.5106544, 69.7326126, -128.8888092, 130.0697937
4: -54.1663132, 76.7551193, -49.3290863, 69.7293549, -123.8956680, 126.0841980

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5501076
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5875135, upper bound: 96.5516410
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -45.6615295, 78.2715302, -41.5644608, 71.5200424, -117.1815567, 119.8359909
1: -49.9450302, 66.0585632, -45.4678535, 60.4032860, -110.3482971, 111.5264130
2: -51.2054443, 66.1545029, -46.6354332, 60.4574471, -111.6628876, 112.7899323
3: -59.0777664, 76.4735184, -53.7209549, 69.9424973, -129.0202637, 130.1944580
4: -54.0997353, 76.6523056, -49.5006561, 69.9765320, -124.0762634, 126.1529388

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5602132
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5875135, upper bound: 96.5638753
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.7164001, 78.4372253, -41.5644608, 71.5200424, -117.2364349, 120.0016861
1: -50.0085220, 66.1290207, -45.4678535, 60.4032860, -110.4118042, 111.5968781
2: -51.2662811, 66.2252960, -46.6354332, 60.4574471, -111.7237167, 112.8607330
3: -59.1561966, 76.5591431, -53.7209549, 69.9424973, -129.0986938, 130.2800903
4: -54.1663132, 76.7551193, -49.5006561, 69.9765320, -124.1428452, 126.2557526

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5496718
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5875135, upper bound: 96.5516410
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -43.6390228, 74.7671432, -45.9628410, 77.8067627, -121.4457550, 120.7299805
1: -47.7354660, 63.1441498, -50.2656403, 66.0334091, -113.7688751, 113.4097900
2: -48.9216156, 63.3015251, -51.4855194, 66.1813202, -115.1029358, 114.7870483
3: -56.3876648, 73.1002960, -59.2746925, 76.5063095, -132.8939514, 132.3749847
4: -51.6179695, 73.2541351, -54.2514763, 76.7217941, -128.3397675, 127.5056152

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5533325, upper bound: 96.5471846
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5638529
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.1196976, 73.8850555, -45.9628410, 77.8067627, -120.9264374, 119.8479004
1: -47.1720428, 62.3058128, -50.2656403, 66.0334091, -113.2054520, 112.5714569
2: -48.3563232, 62.4279175, -51.4855194, 66.1813202, -114.5376434, 113.9134369
3: -55.7405701, 72.1679153, -59.2746925, 76.5063095, -132.2468872, 131.4426117
4: -51.0735397, 72.2635956, -54.2514763, 76.7217941, -127.7953262, 126.5150757

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5533325, upper bound: 96.5471846
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5638529
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -43.6390228, 74.7671432, -45.3854561, 76.8448334, -120.4838562, 120.1525879
1: -47.7354660, 63.1441498, -49.6261864, 65.1088181, -112.8442841, 112.7703400
2: -48.9216156, 63.3015251, -50.8572235, 65.2283630, -114.1499786, 114.1587524
3: -56.3876648, 73.1002960, -58.5323372, 75.4720154, -131.8596649, 131.6326294
4: -51.6179695, 73.2541351, -53.6500931, 75.6274261, -127.2453918, 126.9042282

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5523653, upper bound: 96.5455936
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5753524
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.1196976, 73.8850555, -45.3854561, 76.8448334, -119.9645309, 119.2704926
1: -47.1720428, 62.3058128, -49.6261864, 65.1088181, -112.2808609, 111.9319992
2: -48.3563232, 62.4279175, -50.8572235, 65.2283630, -113.5846863, 113.2851410
3: -55.7405701, 72.1679153, -58.5323372, 75.4720154, -131.2125854, 130.7002563
4: -51.0735397, 72.2635956, -53.6500931, 75.6274261, -126.7009583, 125.9136887

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5523653, upper bound: 96.5455936
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5638529
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -45.0872269, 77.5929565, -45.9628410, 77.8067627, -122.8939667, 123.5558014
1: -49.3714638, 65.5715866, -50.2656403, 66.0334091, -115.4048767, 115.8372192
2: -50.5652313, 65.6822510, -51.4855194, 66.1813202, -116.7465515, 117.1677704
3: -58.4465828, 75.8948364, -59.2746925, 76.5063095, -134.9528961, 135.1695251
4: -53.4376373, 76.0536041, -54.2514763, 76.7217941, -130.1594238, 130.3050842

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5820724, upper bound: 96.5549869
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5869425, upper bound: 96.5869425
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -44.1401176, 76.2326126, -45.9628410, 77.8067627, -121.9468460, 122.1954498
1: -48.3297348, 64.2491226, -50.2656403, 66.0334091, -114.3631439, 114.5147629
2: -49.5412827, 64.3295746, -51.4855194, 66.1813202, -115.7225800, 115.8150940
3: -57.2588310, 74.3965149, -59.2746925, 76.5063095, -133.7651367, 133.6712036
4: -52.4272232, 74.4969254, -54.2514763, 76.7217941, -129.1490021, 128.7483978

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5820724, upper bound: 96.5575128
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5844575, upper bound: 96.5907013
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5837462, upper bound: 96.5868700
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -45.0872269, 77.5929565, -45.3854561, 76.8448334, -121.9320602, 122.9784088
1: -49.3714638, 65.5715866, -49.6261864, 65.1088181, -114.4802856, 115.1977692
2: -50.5652313, 65.6822510, -50.8572235, 65.2283630, -115.7935944, 116.5394745
3: -58.4465828, 75.8948364, -58.5323372, 75.4720154, -133.9185944, 134.4271698
4: -53.4376373, 76.0536041, -53.6500931, 75.6274261, -129.0650482, 129.7037048

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5805017, upper bound: 96.5524210
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5869425, upper bound: 96.6019534
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -44.1401176, 76.2326126, -45.3854561, 76.8448334, -120.9849472, 121.6180573
1: -48.3297348, 64.2491226, -49.6261864, 65.1088181, -113.4385529, 113.8753052
2: -49.5412827, 64.3295746, -50.8572235, 65.2283630, -114.7696228, 115.1867981
3: -57.2588310, 74.3965149, -58.5323372, 75.4720154, -132.7308502, 132.9288483
4: -52.4272232, 74.4969254, -53.6500931, 75.6274261, -128.0546417, 128.1470184

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5805017, upper bound: 96.5549469
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5869425, upper bound: 96.5869425
time: 0.89 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.76 seconds
NS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5269075, upper bound: 96.4664982
NS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5269075, upper bound: 96.4664982
NS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5343734, upper bound: 96.4918933
NS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5343734, upper bound: 96.4918933
NS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5399537, upper bound: 96.4689821
NS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5399543, upper bound: 96.4743218
NS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5473771, upper bound: 96.4943662
NS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5473777, upper bound: 96.4943663
NS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5271949, upper bound: 96.4350133
NS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5426262, upper bound: 96.4641597
NS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5063253, upper bound: 96.4517727
NS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5063253, upper bound: 96.4576761
NS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5041604, upper bound: 96.4404820
NS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5041604, upper bound: 96.4598620
NS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5904166, upper bound: 96.4931919
NS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5894997, upper bound: 96.5023869
NS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5907385, upper bound: 96.5039227
NS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5907385, upper bound: 96.5060198
NS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5512023, upper bound: 96.4971652
NS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5167449, upper bound: 96.4828151
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5923059, upper bound: 96.5049281
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5877020, upper bound: 96.4983632
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5550602, upper bound: 96.4988350
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5550602, upper bound: 96.5007648
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5888331, upper bound: 96.5043087
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5896020, upper bound: 96.5075721
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5365802, upper bound: 96.5365802
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5365802, upper bound: 96.5365802
NS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5252640, upper bound: 96.4903224
NS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5346792, upper bound: 96.5346792
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5376081, upper bound: 96.5453321
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5376081, upper bound: 96.5453316
NS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5395613, upper bound: 96.5456024
NS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5395613, upper bound: 96.5475340
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5474098, upper bound: 96.5383628
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5474098, upper bound: 96.5403354
NS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5511419, upper bound: 96.5391036
NS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5511419, upper bound: 96.5410762
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5491919, upper bound: 96.5491859
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5491919, upper bound: 96.5528045
NS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5529240, upper bound: 96.5499252
NS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5529240, upper bound: 96.5535922
NS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5439511, upper bound: 96.5615244
NS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5439511, upper bound: 96.5615246
NS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5615339
NS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5457850, upper bound: 96.5615341
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5447307, upper bound: 96.5660459
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5447307, upper bound: 96.5660459
NS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5465698, upper bound: 96.5660649
NS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5465698, upper bound: 96.5660649
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5594923, upper bound: 96.5832365
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5594923, upper bound: 96.5840747
NS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5631717, upper bound: 96.5840595
NS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5631717, upper bound: 96.5849430
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5598442, upper bound: 96.5870418
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5598442, upper bound: 96.5880394
NS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5635216, upper bound: 96.5879015
NS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5635216, upper bound: 96.5889178
NS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5496724
NS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5875136, upper bound: 96.5516410
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5501076
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5875135, upper bound: 96.5516410
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5602132
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5875135, upper bound: 96.5638753
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5871351, upper bound: 96.5496718
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5875135, upper bound: 96.5516410
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5533325, upper bound: 96.5471846
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5638529
NS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5533325, upper bound: 96.5471846
NS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5638529
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5523653, upper bound: 96.5455936
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5753524
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5523653, upper bound: 96.5455936
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5557531, upper bound: 96.5638529
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5820724, upper bound: 96.5549869
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5869425, upper bound: 96.5869425
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5844575, upper bound: 96.5907013
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5837462, upper bound: 96.5868700
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5805017, upper bound: 96.5524210
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5869425, upper bound: 96.6019534
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5805017, upper bound: 96.5549469
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 4, lower bound: -96.5869425, upper bound: 96.5869425

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -41.1704483, 71.1371536, -18.0630894, 36.6420746, -77.8125229, 89.2002411
1: -45.0442657, 59.9962807, -19.9198895, 30.0501137, -75.0943680, 79.9161682
2: -46.2192459, 60.0251732, -20.4927845, 29.7470531, -75.9662857, 80.5179443
3: -53.2860374, 69.4535065, -24.0727940, 34.7880821, -88.0741196, 93.5262985
4: -49.0888138, 69.4806747, -23.6052818, 33.8547211, -82.9435349, 93.0859528

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4731098, upper bound: 96.4511583
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5223092, upper bound: 96.4612232
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -41.1704483, 71.1371536, -18.1366940, 36.8058472, -77.9762955, 89.2738495
1: -45.0442657, 59.9962807, -19.9952469, 30.1120872, -75.1563568, 79.9915314
2: -46.2192459, 60.0251732, -20.5636768, 29.8082657, -76.0275116, 80.5888367
3: -53.2860374, 69.4535065, -24.1474533, 34.8551254, -88.1411591, 93.6009598
4: -49.0888138, 69.4806747, -23.6604443, 33.9438667, -83.0326614, 93.1411209

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4731098, upper bound: 96.4511583
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5223092, upper bound: 96.4612232
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.1704483, 71.1371536, -21.8888664, 42.4923439, -83.6627960, 93.0260162
1: -45.0442657, 59.9962807, -24.0269089, 34.9650192, -80.0092773, 84.0231857
2: -46.2192459, 60.0251732, -24.6999092, 34.6418533, -80.8610992, 84.7250824
3: -53.2860374, 69.4535065, -28.8013229, 40.4280815, -93.7141190, 98.2548294
4: -49.0888138, 69.4806747, -27.7145157, 39.5688286, -88.6576233, 97.1951904

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4808282, upper bound: 96.4785682
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5300109, upper bound: 96.4886325
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.1704483, 71.1371536, -21.9817734, 42.7091064, -83.8795547, 93.1189270
1: -45.0442657, 59.9962807, -24.1233158, 35.0707855, -80.1150284, 84.1195984
2: -46.2192459, 60.0251732, -24.7956028, 34.7488213, -80.9680634, 84.8207626
3: -53.2860374, 69.4535065, -28.9028625, 40.5460587, -93.8320923, 98.3563690
4: -49.0888138, 69.4806747, -27.7987823, 39.7096176, -88.7984238, 97.2794571

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4808282, upper bound: 96.4785682
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5300109, upper bound: 96.4886322
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -41.3172379, 71.3980560, -18.0630894, 36.6420746, -77.9593124, 89.4611359
1: -45.2076569, 60.1511192, -19.9198895, 30.0501137, -75.2577591, 80.0710068
2: -46.3758087, 60.1892281, -20.4927845, 29.7470531, -76.1228561, 80.6820145
3: -53.4712257, 69.6379395, -24.0727940, 34.7880821, -88.2593002, 93.7107315
4: -49.2401886, 69.7003784, -23.6052818, 33.8547211, -83.0949097, 93.3056641

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5152361, upper bound: 96.4246495
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5399537, upper bound: 96.4689821
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -41.3172379, 71.3980560, -18.1366940, 36.8058472, -78.1230850, 89.5347443
1: -45.2076569, 60.1511192, -19.9952469, 30.1120872, -75.3197479, 80.1463623
2: -46.3758087, 60.1892281, -20.5636768, 29.8082657, -76.1840744, 80.7529068
3: -53.4712257, 69.6379395, -24.1474533, 34.8551254, -88.3263550, 93.7853928
4: -49.2401886, 69.7003784, -23.6604443, 33.9438667, -83.1840515, 93.3608246

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5152377, upper bound: 96.4246496
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5399543, upper bound: 96.4689821
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.3172379, 71.3980560, -21.8888664, 42.4923439, -83.8095779, 93.2869110
1: -45.2076569, 60.1511192, -24.0269089, 34.9650192, -80.1726608, 84.1780167
2: -46.3758087, 60.1892281, -24.6999092, 34.6418533, -81.0176620, 84.8891373
3: -53.4712257, 69.6379395, -28.8013229, 40.4280815, -93.8992996, 98.4392624
4: -49.2401886, 69.7003784, -27.7145157, 39.5688286, -88.8090134, 97.4148865

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5430207, upper bound: 96.4909912
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5433382, upper bound: 96.4910644
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.3172379, 71.3980560, -21.9817734, 42.7091064, -84.0263443, 93.3798218
1: -45.2076569, 60.1511192, -24.1233158, 35.0707855, -80.2784348, 84.2744370
2: -46.3758087, 60.1892281, -24.7956028, 34.7488213, -81.1246338, 84.9848328
3: -53.4712257, 69.6379395, -28.9028625, 40.5460587, -94.0172806, 98.5408020
4: -49.2401886, 69.7003784, -27.7987823, 39.7096176, -88.9498062, 97.4991608

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5430213, upper bound: 96.4909913
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5433387, upper bound: 96.4910639
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -41.9352303, 72.2366638, -18.3364868, 35.9809952, -77.9162064, 90.5731506
1: -45.8540802, 60.7566032, -20.1459961, 29.9495640, -75.8036346, 80.9026031
2: -47.0273743, 60.8546638, -20.7173233, 29.6063004, -76.6336670, 81.5719910
3: -54.1787529, 70.3179169, -24.1973495, 34.7193451, -88.8980789, 94.5152512
4: -49.6724319, 70.4120941, -23.8477745, 33.7629471, -83.4353790, 94.2598648

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -41.9352303, 72.2366638, -18.7592087, 37.0195541, -78.9547577, 90.9958725
1: -45.8540802, 60.7566032, -20.5995636, 30.6442680, -76.4983444, 81.3561707
2: -47.0273743, 60.8546638, -21.2169590, 30.3305225, -77.3578949, 82.0716248
3: -54.1787529, 70.3179169, -24.7846031, 35.4948807, -89.6736145, 95.1025085
4: -49.6724319, 70.4120941, -24.2572594, 34.6226234, -84.2950516, 94.6693573

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.7973251, 75.6151886, -20.6217041, 40.7382851, -84.5356064, 96.2368927
1: -47.9171906, 63.6562996, -22.6619263, 33.3460236, -81.2632141, 86.3181915
2: -49.1294861, 63.7237816, -23.2862854, 33.0206947, -82.1501617, 87.0100708
3: -56.7174110, 73.6727219, -27.2137947, 38.5237274, -95.2411346, 100.8865128
4: -52.0038490, 73.7733383, -26.3179741, 37.6551971, -89.6590424, 100.0913086

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5812236, upper bound: 96.5050629
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5791169, upper bound: 96.5039354
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -43.7973251, 75.6151886, -20.7368050, 40.9790268, -84.7763290, 96.3519897
1: -47.9171906, 63.6562996, -22.7814178, 33.4775543, -81.3947449, 86.4376907
2: -49.1294861, 63.7237816, -23.4065323, 33.1530991, -82.2825851, 87.1303101
3: -56.7174110, 73.6727219, -27.3401146, 38.6730843, -95.3904724, 101.0128326
4: -52.0038490, 73.7733383, -26.4298267, 37.8236313, -89.8274765, 100.2031631

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5794619, upper bound: 96.5017531
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5763122, upper bound: 96.5004630
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -43.7973251, 75.6151886, -20.8014870, 39.7230301, -83.5203323, 96.4166718
1: -47.9171906, 63.6562996, -22.7817822, 32.9089890, -80.8261795, 86.4380493
2: -49.1294861, 63.7237816, -23.4474792, 32.6521454, -81.7816315, 87.1712570
3: -56.7174110, 73.6727219, -27.2743435, 38.0748711, -94.7922821, 100.9470673
4: -52.0038490, 73.7733383, -26.2826500, 37.2585182, -89.2623672, 100.0559845

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5827039, upper bound: 96.5036166
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809851, upper bound: 96.5029873
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -43.7973251, 75.6151886, -22.5886326, 42.5229568, -86.3202667, 98.2038193
1: -47.9171906, 63.6562996, -24.7539234, 35.3554688, -83.2726593, 88.4101791
2: -49.1294861, 63.7237816, -25.4524117, 35.0863953, -84.2158737, 89.1761932
3: -56.7174110, 73.6727219, -29.6504688, 40.9273987, -97.6448059, 103.3231888
4: -52.0038490, 73.7733383, -28.3849220, 40.1846161, -92.1884613, 102.1582642

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5827039, upper bound: 96.5057066
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809851, upper bound: 96.5049169
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -45.2390633, 75.7938309, -19.9438782, 39.7412262, -84.9802780, 95.7377014
1: -49.4263840, 64.4019470, -21.9242134, 32.4363785, -81.8627625, 86.3261566
2: -50.6609344, 64.6012115, -22.5254211, 32.1170845, -82.7780151, 87.1266251
3: -58.3063240, 74.6300278, -26.3384438, 37.4642639, -95.7705841, 100.9684677
4: -53.3483582, 74.9629974, -25.5686646, 36.5892029, -89.9375534, 100.5316620

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5469062, upper bound: 96.4971323
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5389908, upper bound: 96.4951997
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.1301308, 77.8321686, -19.9438782, 39.7412262, -85.8713455, 97.7760391
1: -50.4351044, 66.0740509, -21.9242134, 32.4363785, -82.8714600, 87.9982605
2: -51.6798859, 66.2257919, -22.5254211, 32.1170845, -83.7969666, 88.7512131
3: -59.6054382, 76.5463333, -26.3384438, 37.4642639, -97.0697021, 102.8847733
4: -54.5248718, 76.8399048, -25.5686646, 36.5892029, -91.1140518, 102.4085693

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5689486, upper bound: 96.5021802
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5889451, upper bound: 96.5048552
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.1081047, 77.8020325, -23.3340416, 44.4658966, -90.5740051, 101.1360779
1: -50.4112625, 66.0476074, -25.6164551, 37.0544815, -87.4657211, 91.6640625
2: -51.6556015, 66.1987610, -26.3607426, 36.7374344, -88.3930359, 92.5595016
3: -59.5781136, 76.5155640, -30.7477608, 42.8721886, -102.4502869, 107.2633209
4: -54.5011520, 76.8073273, -29.5787964, 42.0464630, -96.5476151, 106.3861237

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5076950, upper bound: 96.4511172
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5847677, upper bound: 96.4983632
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -45.2390633, 75.7938309, -20.8014870, 39.7230301, -84.9620895, 96.5953064
1: -49.4263840, 64.4019470, -22.7817822, 32.9089890, -82.3353729, 87.1837158
2: -50.6609344, 64.6012115, -23.4474792, 32.6521454, -83.3130798, 88.0486755
3: -58.3063240, 74.6300278, -27.2743435, 38.0748711, -96.3811951, 101.9043655
4: -53.3483582, 74.9629974, -26.2826500, 37.2585182, -90.6068726, 101.2456436

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5493458, upper bound: 96.4975277
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5548028, upper bound: 96.4985507
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -45.2390633, 75.7938309, -22.5886326, 42.5229568, -87.7620239, 98.3824615
1: -49.4263840, 64.4019470, -24.7539234, 35.3554688, -84.7818527, 89.1558456
2: -50.6609344, 64.6012115, -25.4524117, 35.0863953, -85.7473297, 90.0536194
3: -58.3063240, 74.6300278, -29.6504688, 40.9273987, -99.2337189, 104.2804947
4: -53.3483582, 74.9629974, -28.3849220, 40.1846161, -93.5329742, 103.3479156

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5493458, upper bound: 96.4994702
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5548028, upper bound: 96.5004935
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.1301308, 77.8321686, -22.1205254, 41.6168289, -87.7469559, 99.9526825
1: -50.4351044, 66.0740509, -24.2150688, 34.6078949, -85.0429840, 90.2891235
2: -51.6798859, 66.2257919, -24.8844643, 34.3812943, -86.0611649, 91.1102600
3: -59.6054382, 76.5463333, -28.9171581, 40.0275955, -99.6330261, 105.4634857
4: -54.5248718, 76.8399048, -27.5637703, 39.2973137, -93.8221893, 104.4036713

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5722743, upper bound: 96.5024161
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5722743, upper bound: 96.5043087
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -46.1301308, 77.8321686, -21.5238342, 40.7519913, -86.8821259, 99.3559952
1: -50.4351044, 66.0740509, -23.5900841, 33.7906647, -84.2257385, 89.6641388
2: -51.6798859, 66.2257919, -24.2422447, 33.5059280, -85.1858139, 90.4680328
3: -59.6054382, 76.5463333, -28.2025795, 39.1401901, -98.7456131, 104.7489166
4: -54.5248718, 76.8399048, -27.0026112, 38.3132095, -92.8380814, 103.8425140

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5724655, upper bound: 96.5055474
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5724655, upper bound: 96.5075721
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -41.1540489, 70.8628464, -112.0168915, 112.0168762
1: -45.0143814, 59.8876724, -45.0143814, 59.8876724, -104.9020538, 104.9020538
2: -46.1856041, 59.9343796, -46.1856041, 59.9343796, -106.1199799, 106.1199799
3: -53.1999969, 69.3314743, -53.1999969, 69.3314743, -122.5314713, 122.5314636
4: -49.0350761, 69.3348389, -49.0350761, 69.3348389, -118.3699036, 118.3699036

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -41.1126022, 70.0654449, -111.2194901, 111.9754486
1: -45.0143814, 59.8876724, -44.9368896, 59.3829651, -104.3973465, 104.8245621
2: -46.1856041, 59.9343796, -46.0977135, 59.4603043, -105.6459045, 106.0320892
3: -53.1999969, 69.3314743, -53.0495338, 68.7801971, -121.9801941, 122.3810043
4: -49.0350761, 69.3348389, -48.8637085, 68.8170929, -117.8521576, 118.1985474

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4841069, upper bound: 96.5052521
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5322046, upper bound: 96.5342001
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -39.7030029, 68.9658813, -110.0784836, 109.7684479
1: -44.9368896, 59.3829651, -43.4227562, 58.1375275, -103.0744019, 102.8057251
2: -46.0977135, 59.4603043, -44.5911942, 58.1383896, -104.2360992, 104.0514984
3: -53.0495338, 68.7801971, -51.3604164, 67.2750626, -120.3245926, 120.1406097
4: -48.8637085, 68.8170929, -47.4538345, 67.1721039, -116.0358124, 116.2709198

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4809072, upper bound: 96.4809072
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4809072, upper bound: 96.4903224
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -41.1724625, 70.9466400, -112.0592346, 111.2379074
1: -44.9368896, 59.3829651, -45.0366440, 59.9487305, -104.8856201, 104.4196091
2: -46.0977135, 59.4603043, -46.2079086, 59.9872284, -106.0849457, 105.6682053
3: -53.0495338, 68.7801971, -53.2251320, 69.4066467, -122.4561768, 122.0053253
4: -48.8637085, 68.8170929, -49.0748672, 69.3876190, -118.2513275, 117.8919449

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4841069, upper bound: 96.5010687
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4841069, upper bound: 96.5346792
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -41.3208656, 71.1446304, -112.2986603, 112.1837158
1: -45.0143814, 59.8876724, -45.1999130, 60.0637360, -105.0781021, 105.0875854
2: -46.1856041, 59.9343796, -46.3642349, 60.1203766, -106.3059845, 106.2986145
3: -53.1999969, 69.3314743, -53.4104080, 69.5413132, -122.7413101, 122.7418823
4: -49.0350761, 69.3348389, -49.2065659, 69.5821609, -118.6172333, 118.5414047

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4861838, upper bound: 96.5142515
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5333086, upper bound: 96.5408647
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.1540489, 70.8628464, -41.2772064, 70.3409424, -111.4949799, 112.1400375
1: -45.0143814, 59.8876724, -45.1196556, 59.5536995, -104.5680847, 105.0073242
2: -46.1856041, 59.9343796, -46.2735367, 59.6405334, -105.8261414, 106.2079163
3: -53.1999969, 69.3314743, -53.2558594, 68.9840393, -122.1840363, 122.5873032
4: -49.0350761, 69.3348389, -49.0312271, 69.0583344, -118.0934067, 118.3660507

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4861838, upper bound: 96.5142515
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5333086, upper bound: 96.5431151
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -41.3208656, 71.1446304, -112.2572250, 111.3863068
1: -44.9368896, 59.3829651, -45.1999130, 60.0637360, -105.0006104, 104.5828781
2: -46.0977135, 59.4603043, -46.3642349, 60.1203766, -106.2180862, 105.8245392
3: -53.0495338, 68.7801971, -53.4104080, 69.5413132, -122.5908508, 122.1906052
4: -48.8637085, 68.8170929, -49.2065659, 69.5821609, -118.4458694, 118.0236588

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4870789, upper bound: 96.5121157
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5352856, upper bound: 96.5412242
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -41.1126022, 70.0654449, -41.2772064, 70.3409424, -111.4535446, 111.3426437
1: -44.9368896, 59.3829651, -45.1196556, 59.5536995, -104.4905853, 104.5026169
2: -46.0977135, 59.4603043, -46.2735367, 59.6405334, -105.7382355, 105.7338409
3: -53.0495338, 68.7801971, -53.2558594, 68.9840393, -122.0335693, 122.0360489
4: -48.8637085, 68.8170929, -49.0312271, 69.0583344, -117.9220428, 117.8482971

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4870789, upper bound: 96.5121157
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5352857, upper bound: 96.5432244
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -41.1540489, 70.8628464, -112.1837082, 112.2986450
1: -45.1999130, 60.0637360, -45.0143814, 59.8876724, -105.0875854, 105.0780945
2: -46.3642349, 60.1203766, -46.1856041, 59.9343796, -106.2986145, 106.3059845
3: -53.4104080, 69.5413132, -53.1999969, 69.3314743, -122.7418823, 122.7413025
4: -49.2065659, 69.5821609, -49.0350761, 69.3348389, -118.5413971, 118.6172333

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5428553, upper bound: 96.5338811
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5431375, upper bound: 96.5340307
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -41.1126022, 70.0654449, -111.3863068, 112.2572250
1: -45.1999130, 60.0637360, -44.9368896, 59.3829651, -104.5828781, 105.0006104
2: -46.3642349, 60.1203766, -46.0977135, 59.4603043, -105.8245392, 106.2180939
3: -53.4104080, 69.5413132, -53.0495338, 68.7801971, -122.1906052, 122.5908432
4: -49.2065659, 69.5821609, -48.8637085, 68.8170929, -118.0236511, 118.4458694

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5428553, upper bound: 96.5358766
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5431375, upper bound: 96.5360262
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -41.1540489, 70.8628464, -112.1400528, 111.4949799
1: -45.1196556, 59.5536995, -45.0143814, 59.8876724, -105.0073242, 104.5680847
2: -46.2735367, 59.6405334, -46.1856041, 59.9343796, -106.2079163, 105.8261414
3: -53.2558594, 68.9840393, -53.1999969, 69.3314743, -122.5873184, 122.1840363
4: -49.0312271, 69.0583344, -49.0350761, 69.3348389, -118.3660431, 118.0934143

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5469584, upper bound: 96.5346697
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5470147, upper bound: 96.5347773
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -41.1126022, 70.0654449, -111.3426514, 111.4535446
1: -45.1196556, 59.5536995, -44.9368896, 59.3829651, -104.5026093, 104.4905853
2: -46.2735367, 59.6405334, -46.0977135, 59.4603043, -105.7338409, 105.7382355
3: -53.2558594, 68.9840393, -53.0495338, 68.7801971, -122.0360413, 122.0335693
4: -49.0312271, 69.0583344, -48.8637085, 68.8170929, -117.8482971, 117.9220428

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5469584, upper bound: 96.5366651
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5470147, upper bound: 96.5347773
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -41.3208656, 71.1446304, -112.4654922, 112.4654846
1: -45.1999130, 60.0637360, -45.1999130, 60.0637360, -105.2636337, 105.2636337
2: -46.3642349, 60.1203766, -46.3642349, 60.1203766, -106.4846115, 106.4846115
3: -53.4104080, 69.5413132, -53.4104080, 69.5413132, -122.9517212, 122.9517212
4: -49.2065659, 69.5821609, -49.2065659, 69.5821609, -118.7887268, 118.7887268

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5446810, upper bound: 96.5448135
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5449502, upper bound: 96.5449632
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -41.3208656, 71.1446304, -41.2772064, 70.3409424, -111.6618042, 112.4218216
1: -45.1999130, 60.0637360, -45.1196556, 59.5536995, -104.7536163, 105.1833649
2: -46.3642349, 60.1203766, -46.2735367, 59.6405334, -106.0047531, 106.3939133
3: -53.4104080, 69.5413132, -53.2558594, 68.9840393, -122.3944473, 122.7971573
4: -49.2065659, 69.5821609, -49.0312271, 69.0583344, -118.2649002, 118.6133652

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5446810, upper bound: 96.5486735
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5449502, upper bound: 96.5449632
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -41.3208656, 71.1446304, -112.4218292, 111.6618042
1: -45.1196556, 59.5536995, -45.1999130, 60.0637360, -105.1833572, 104.7536163
2: -46.2735367, 59.6405334, -46.3642349, 60.1203766, -106.3939133, 106.0047684
3: -53.2558594, 68.9840393, -53.4104080, 69.5413132, -122.7971649, 122.3944473
4: -49.0312271, 69.0583344, -49.2065659, 69.5821609, -118.6133652, 118.2649002

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5487840, upper bound: 96.5456021
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5488404, upper bound: 96.5457098
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -41.2772064, 70.3409424, -41.2772064, 70.3409424, -111.6181488, 111.6181412
1: -45.1196556, 59.5536995, -45.1196556, 59.5536995, -104.6733398, 104.6733398
2: -46.2735367, 59.6405334, -46.2735367, 59.6405334, -105.9140701, 105.9140701
3: -53.2558594, 68.9840393, -53.2558594, 68.9840393, -122.2398911, 122.2398911
4: -49.0312271, 69.0583344, -49.0312271, 69.0583344, -118.0895538, 118.0895462

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5487841, upper bound: 96.5456021
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5488404, upper bound: 96.5457098
time: 1.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.43 + 418.05 = 420.48 seconds
