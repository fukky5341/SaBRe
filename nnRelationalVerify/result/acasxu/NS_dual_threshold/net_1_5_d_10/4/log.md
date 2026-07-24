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
execution time: IAR + RelationalAnalysis = 0.64 + 2.34 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2372775, upper bound: 60.2372775

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2262360, upper bound: 60.2331846
time: 0.89 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2372756, upper bound: 60.2372756
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.82 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 4, lower bound: -60.2262360, upper bound: 60.2331846
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 4, lower bound: -60.2372756, upper bound: 60.2372756

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -12.1294031, 37.7388992, -11.7758732, 35.8175621, -47.9469643, 49.5147667
1: -17.1822987, 39.1265984, -16.6260567, 37.2756233, -54.4579239, 55.7526474
2: -14.7555904, 43.5125732, -14.3642073, 41.4520645, -56.2076569, 57.8767776
3: -16.1523533, 55.9294815, -15.5200558, 53.0719376, -69.2242889, 71.4495087
4: -13.7831745, 51.7584686, -13.4019213, 49.3027229, -63.0858917, 65.1603928

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2164519
time: 0.81 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1809815, upper bound: 60.1913764
time: 0.78 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -12.1294031, 37.7388992, -12.0028152, 37.3610191, -49.4904213, 49.7417068
1: -17.1822987, 39.1265984, -17.0043507, 38.7351952, -55.9174957, 56.1309509
2: -14.7555904, 43.5125732, -14.6032085, 43.0774536, -57.8330460, 58.1157837
3: -16.1523533, 55.9294815, -15.9831753, 55.3704910, -71.5228424, 71.9126511
4: -13.7831745, 51.7584686, -13.6444702, 51.2401848, -65.0233612, 65.4029236

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2332555
time: 0.87 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412
time: 0.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.43 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2164519
NS_B1_B2, status: Status.VERIFIED, split count: 2, time: 2.43
Output dim: 4, lower bound: -60.1809815, upper bound: 60.1913764
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 4, lower bound: -60.2153461, upper bound: 60.2332555
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -11.9775000, 37.3109016, -9.6363459, 29.7178612, -41.6953621, 46.9472427
1: -16.9736977, 38.6834106, -13.6850090, 30.9666481, -47.9403458, 52.3684196
2: -14.5771618, 43.0234413, -11.8295956, 34.4676971, -49.0448494, 54.8530350
3: -15.9551287, 55.3119316, -12.7319603, 44.2135086, -60.1686363, 68.0438919
4: -13.6242332, 51.1773186, -11.1341496, 40.9912453, -54.6154785, 62.3114700

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1809815, upper bound: 60.1913764
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1809815, upper bound: 60.1913764
time: 0.84 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -11.9775000, 37.3109016, -10.3209095, 32.6361275, -44.6136284, 47.6318054
1: -16.9736977, 38.6834106, -14.6892014, 33.8541946, -50.8278885, 53.3726082
2: -14.5771618, 43.0234413, -12.6255856, 37.6878052, -52.2649689, 55.6490250
3: -15.9551287, 55.3119316, -13.8037653, 48.5566101, -64.5117416, 69.1156921
4: -13.6242332, 51.1773186, -11.8977566, 44.8329659, -58.4571991, 63.0750732

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412
time: 0.85 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412
time: 0.80 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -11.5280428, 36.0634918, -11.2069159, 35.2389679, -46.7670097, 47.2704086
1: -16.3616581, 37.3929443, -15.8723192, 36.5255699, -52.8872299, 53.2652626
2: -14.0542173, 41.5972443, -13.6602526, 40.7018242, -54.7560425, 55.2574959
3: -15.3825579, 53.4967842, -14.9416742, 52.3413239, -67.7238846, 68.4384613
4: -13.1590786, 49.4717827, -12.8325424, 48.3522224, -61.5112991, 62.3043251

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1809815, upper bound: 60.2304412
time: 0.80 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1809815, upper bound: 60.2304412
time: 0.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.40 seconds
NS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.40
Output dim: 4, lower bound: -60.1809815, upper bound: 60.1913764
NS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.40
Output dim: 4, lower bound: -60.1809815, upper bound: 60.1913764
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 4, lower bound: -60.1809815, upper bound: 60.2304412
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 4, lower bound: -60.1809815, upper bound: 60.2304412

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -10.4466705, 33.0043793, -10.3209095, 32.6361275, -43.0827942, 43.3252754
1: -14.8643341, 34.2352486, -14.6892014, 33.8541946, -48.7185211, 48.9244499
2: -12.7760410, 38.1106606, -12.6255856, 37.6878052, -50.4638443, 50.7362442
3: -13.9698524, 49.0998573, -13.8037653, 48.5566101, -62.5264626, 62.9036217
4: -12.0348778, 45.3377380, -11.8977566, 44.8329659, -56.8678360, 57.2354965

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2238652
time: 0.83 seconds

## Relational analysis of NS_B2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2332555
time: 0.94 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -11.5646820, 36.3494148, -10.3209095, 32.6361275, -44.2008095, 46.6703186
1: -16.3725929, 37.6687965, -14.6892014, 33.8541946, -50.2267838, 52.3579979
2: -14.0860319, 41.9790382, -12.6255856, 37.6878052, -51.7738380, 54.6046219
3: -15.4223356, 53.9925766, -13.8037653, 48.5566101, -63.9789467, 67.7963409
4: -13.2208614, 49.8709221, -11.8977566, 44.8329659, -58.0538254, 61.7686768

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2238652
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2332555
time: 0.85 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -10.4466705, 33.0043793, -11.2069159, 35.2389679, -45.6856270, 44.2112923
1: -14.8643341, 34.2352486, -15.8723192, 36.5255699, -51.3899040, 50.1075630
2: -12.7760410, 38.1106606, -13.6602526, 40.7018242, -53.4778671, 51.7709122
3: -13.9698524, 49.0998573, -14.9416742, 52.3413239, -66.3111649, 64.0415268
4: -12.0348778, 45.3377380, -12.8325424, 48.3522224, -60.3870964, 58.1702805

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098459, upper bound: 60.2230902
time: 0.92 seconds

## Relational analysis of NS_B2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233603, upper bound: 60.2233604
time: 0.79 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -11.5646820, 36.3494148, -11.2069159, 35.2389679, -46.8036499, 47.5563316
1: -16.3725929, 37.6687965, -15.8723192, 36.5255699, -52.8981628, 53.5411110
2: -14.0860319, 41.9790382, -13.6602526, 40.7018242, -54.7878494, 55.6392899
3: -15.4223356, 53.9925766, -14.9416742, 52.3413239, -67.7636566, 68.9342499
4: -13.2208614, 49.8709221, -12.8325424, 48.3522224, -61.5730820, 62.7034645

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2210875
time: 1.04 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412
time: 1.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.79 seconds
NS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2238652
NS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2332555
NS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2238652
NS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2323385, upper bound: 60.2332555
NS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2098459, upper bound: 60.2230902
NS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2233603, upper bound: 60.2233604
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2210875
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 4, lower bound: -60.2304412, upper bound: 60.2304412

## BFS NS instance: NS_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -10.2003736, 32.2737617, -40.9729347, 37.9370537
1: -12.4092007, 28.8249302, -14.5195646, 33.4811172, -45.8903160, 43.3444939
2: -10.6860304, 32.1388817, -12.4810371, 37.2763100, -47.9623413, 44.6199150
3: -11.6377096, 41.3525887, -13.6422672, 48.0248795, -59.6625824, 54.9948578
4: -10.1306763, 38.2451363, -11.7653694, 44.3450813, -54.4757576, 50.0105057

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
time: 0.97 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
time: 0.89 seconds

## BFS NS instance: NS_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -16.5652657, 49.3458900, -10.2198400, 32.3301277, -48.8953934, 59.5657272
1: -23.0251637, 51.0942726, -14.5454531, 33.5391541, -56.5643120, 65.6397247
2: -19.7440033, 56.8110428, -12.5035725, 37.3385086, -57.0825119, 69.3146133
3: -21.7395668, 72.8528137, -13.6702290, 48.1051750, -69.8447418, 86.5230408
4: -18.1693535, 67.7313232, -11.7880955, 44.4159164, -62.5852699, 79.5194168

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2340061, upper bound: 60.2366528
time: 1.06 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2360914, upper bound: 60.2360914
time: 0.91 seconds

## BFS NS instance: NS_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -10.2003736, 32.2737617, -42.2249489, 41.7485695
1: -14.1082268, 32.7253380, -14.5195646, 33.4811172, -47.5893402, 47.2449036
2: -12.1443062, 36.5262375, -12.4810371, 37.2763100, -49.4206123, 49.0072708
3: -13.2896729, 46.9455948, -13.6422672, 48.0248795, -61.3145523, 60.5878601
4: -11.4511833, 43.4067154, -11.7653694, 44.3450813, -55.7962570, 55.1720848

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2238652
time: 1.13 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2238652
time: 1.13 seconds

## BFS NS instance: NS_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -10.2198400, 32.3301277, -48.4494247, 58.9589920
1: -22.4673023, 50.3988075, -14.5454531, 33.5391541, -56.0064545, 64.9442596
2: -19.2585125, 56.0775299, -12.5035725, 37.3385086, -56.5970192, 68.5810928
3: -21.2393341, 71.9838791, -13.6702290, 48.1051750, -69.3445129, 85.6541061
4: -17.8179779, 66.7789383, -11.7880955, 44.4159164, -62.2338943, 78.5670242

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2332555
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2332555
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.1880865, 32.2393837, -7.8912673, 25.4425735, -35.6306610, 40.1306496
1: -14.4970589, 33.4476166, -11.1526108, 26.4303532, -40.9274139, 44.6002274
2: -12.4596758, 37.2423172, -9.5916300, 29.5862541, -42.0459290, 46.8339462
3: -13.6287451, 47.9754448, -10.6004381, 37.9559097, -51.5846519, 58.5758820
4: -11.7495270, 44.3038292, -9.1747046, 35.1177864, -46.8673096, 53.4785347

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2112850
time: 0.82 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2254849
time: 0.91 seconds

## BFS NS instance: NS_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.1517124, 32.2116241, -11.5309305, 36.2892380, -46.4409447, 43.7425537
1: -14.4553766, 33.4170647, -16.3129578, 37.6009598, -52.0563354, 49.7300186
2: -12.4214363, 37.2063599, -14.0159235, 41.8694153, -54.2908516, 51.2222824
3: -13.6020317, 47.9441528, -15.3604355, 53.7925797, -67.3946075, 63.3045883
4: -11.7227383, 44.2546272, -13.1525908, 49.7282600, -61.4509964, 57.4072075

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1994042
time: 1.02 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005528
time: 0.79 seconds

## BFS NS instance: NS_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -11.0984592, 34.9182053, -44.8693962, 42.6466522
1: -14.1082268, 32.7253380, -15.7181921, 36.1946487, -50.3028717, 48.4435272
2: -12.1443062, 36.5262375, -13.5282660, 40.3376884, -52.4819946, 50.0545044
3: -13.2896729, 46.9455948, -14.7972155, 51.8713493, -65.1610107, 61.7428093
4: -11.4511833, 43.4067154, -12.7124605, 47.9204750, -59.3716545, 56.1191711

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 0.85 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 0.89 seconds

## BFS NS instance: NS_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -11.0362692, 34.7292824, -50.8485718, 59.7754211
1: -22.4673023, 50.3988075, -15.6410265, 35.9980659, -58.4653702, 66.0398331
2: -19.2585125, 56.0775299, -13.4604273, 40.1105614, -59.3690681, 69.5379410
3: -21.2393341, 71.9838791, -14.7219524, 51.5907631, -72.8300858, 86.7058334
4: -17.8179779, 66.7789383, -12.6530437, 47.6573868, -65.4753647, 79.4319611

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098459, upper bound: 60.2230902
time: 0.89 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233603, upper bound: 60.2233604
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.68 seconds
NS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
NS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2352937, upper bound: 60.2352937
NS_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2340061, upper bound: 60.2366528
NS_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2360914, upper bound: 60.2360914
NS_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2238652
NS_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2238652
NS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2332555
NS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2298993, upper bound: 60.2332555
NS_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2112850
NS_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2254849
NS_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1994042
NS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005528
NS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
NS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
NS_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2098459, upper bound: 60.2230902
NS_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 4, lower bound: -60.2233603, upper bound: 60.2233604

## BFS NS instance: NS_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -8.5392914, 27.2805252, -35.9796982, 36.2759705
1: -12.4092007, 28.8249302, -12.1856346, 28.3512688, -40.7604675, 41.0105667
2: -10.6860304, 32.1388817, -10.4923859, 31.6131878, -42.2992172, 42.6312675
3: -11.6377096, 41.3525887, -11.4304800, 40.6814766, -52.3191833, 52.7830696
4: -10.1306763, 38.2451363, -9.9563656, 37.6183701, -47.7490463, 48.2014923

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185508, upper bound: 60.2258015
time: 0.87 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319440, upper bound: 60.2319440
time: 1.04 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6991749, 27.7366791, -16.3947735, 48.7803421, -57.4795151, 44.1314545
1: -12.4092007, 28.8249302, -22.7656269, 50.5103569, -62.9195518, 51.5905571
2: -10.6860304, 32.1388817, -19.5160103, 56.1796188, -66.8656464, 51.6548805
3: -11.6377096, 41.3525887, -21.4941235, 72.0030670, -83.6407776, 62.8467102
4: -10.1306763, 38.2451363, -17.9616222, 66.9744339, -77.1051102, 56.2067490

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290673, upper bound: 60.2193800
time: 1.05 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319440, upper bound: 60.2319440
time: 1.02 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -15.0562019, 45.0254517, -10.0520954, 31.8484936, -46.9046936, 55.0775452
1: -20.8223305, 46.6315765, -14.3108797, 33.0433273, -53.8656502, 60.9424553
2: -17.9141350, 51.9317551, -12.3050194, 36.7915268, -54.7056541, 64.2367554
3: -19.7496395, 66.6583252, -13.4510946, 47.4107018, -67.1603394, 80.1094055
4: -16.6299400, 61.9274216, -11.6149111, 43.7690353, -60.3989754, 73.5423355

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2173488, upper bound: 60.2301730
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334608, upper bound: 60.2349806
time: 0.88 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -16.5011749, 49.2117462, -10.2198400, 32.3301277, -48.8313026, 59.4315872
1: -22.9424286, 50.9540787, -14.5454531, 33.5391541, -56.4815750, 65.4995193
2: -19.6721401, 56.6566772, -12.5035725, 37.3385086, -57.0106430, 69.1602478
3: -21.6692696, 72.6607208, -13.6702290, 48.1051750, -69.7744446, 86.3309479
4: -18.1088314, 67.5434189, -11.7880955, 44.4159164, -62.5247498, 79.3315048

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180274, upper bound: 60.2301540
time: 0.97 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2341204, upper bound: 60.2341204
time: 0.88 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -8.5392914, 27.2805252, -37.2317162, 40.0874825
1: -14.1082268, 32.7253380, -12.1856346, 28.3512688, -42.4594955, 44.9109726
2: -12.1443062, 36.5262375, -10.4923859, 31.6131878, -43.7574921, 47.0186234
3: -13.2896729, 46.9455948, -11.4304800, 40.6814766, -53.9711494, 58.3760757
4: -11.4511833, 43.4067154, -9.9563656, 37.6183701, -49.0695457, 53.3630753

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202114, upper bound: 60.2155934
time: 1.02 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_B1_B1

### Relational analysis result of NS_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2088942, upper bound: 60.2004083
time: 0.96 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274246, upper bound: 60.2197356
time: 0.91 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2295365, upper bound: 60.2226549
time: 0.99 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -16.3947735, 48.7803421, -58.7315331, 47.9429703
1: -14.1082268, 32.7253380, -22.7656269, 50.5103569, -64.6185760, 55.4909668
2: -12.1443062, 36.5262375, -19.5160103, 56.1796188, -68.3239136, 56.0422401
3: -13.2896729, 46.9455948, -21.4941235, 72.0030670, -85.2927246, 68.4397202
4: -11.4511833, 43.4067154, -17.9616222, 66.9744339, -78.4256134, 61.3683319

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202114, upper bound: 60.2155934
time: 0.92 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274246, upper bound: 60.2197356
time: 0.91 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2295365, upper bound: 60.2226549
time: 0.90 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -8.5392914, 27.2805252, -43.3998184, 57.2784386
1: -22.4673023, 50.3988075, -12.1856346, 28.3512688, -50.8185730, 62.5844383
2: -19.2585125, 56.0775299, -10.4923859, 31.6131878, -50.8716927, 66.5699081
3: -21.2393341, 71.9838791, -11.4304800, 40.6814766, -61.9208107, 83.4143600
4: -17.8179779, 66.7789383, -9.9563656, 37.6183701, -55.4363480, 76.7352982

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2107919, upper bound: 60.2249681
time: 0.93 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2220471, upper bound: 60.2293598
time: 0.99 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -16.3947735, 48.7803421, -64.8996429, 65.1339111
1: -22.4673023, 50.3988075, -22.7656269, 50.5103569, -72.9776611, 73.1644287
2: -19.2585125, 56.0775299, -19.5160103, 56.1796188, -75.4381332, 75.5935364
3: -21.2393341, 71.9838791, -21.4941235, 72.0030670, -93.2423935, 93.4780045
4: -17.8179779, 66.7789383, -17.9616222, 66.9744339, -84.7924118, 84.7405624

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2297509, upper bound: 60.2332555
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2260046, upper bound: 60.2332105
time: 1.05 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.2697396, 23.5177422, -7.8912673, 25.4425735, -32.7123108, 31.4090099
1: -10.3344593, 24.4534683, -11.1526108, 26.4303532, -36.7648048, 35.6060715
2: -8.8746996, 27.3351669, -9.5916300, 29.5862541, -38.4609528, 36.9267960
3: -9.7782707, 35.1258240, -10.6004381, 37.9559097, -47.7341805, 45.7262611
4: -8.5222330, 32.4922867, -9.1747046, 35.1177864, -43.6400146, 41.6669884

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2123232
time: 0.79 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2142216, upper bound: 60.2118209
time: 0.83 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.5560017, 36.1935806, -7.8912673, 25.4425735, -36.9985733, 44.0848465
1: -16.3586349, 37.5258446, -11.1526108, 26.4303532, -42.7889862, 48.6784515
2: -14.0466652, 41.7448120, -9.5916300, 29.5862541, -43.6329117, 51.3364410
3: -15.4071274, 53.6530113, -10.6004381, 37.9559097, -53.3630371, 64.2534485
4: -13.1645775, 49.6421280, -9.1747046, 35.1177864, -48.2823601, 58.8168297

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2229628
time: 1.13 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2142216, upper bound: 60.2118209
time: 1.28 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.5676270, 30.4375210, -11.5309305, 36.2892380, -45.8568611, 41.9684525
1: -13.6437168, 31.5979614, -16.3129578, 37.6009598, -51.2446747, 47.9109116
2: -11.7235870, 35.2045860, -14.0159235, 41.8694153, -53.5930023, 49.2205086
3: -12.8219328, 45.3334160, -15.3604355, 53.7925797, -66.6145096, 60.6938515
4: -11.0839100, 41.8613129, -13.1525908, 49.7282600, -60.8121719, 55.0138931

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210187, upper bound: 60.1916621
time: 1.19 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202904, upper bound: 60.1994042
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1958138
time: 0.97 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.6414785, 42.3319702, -11.3979406, 35.9310875, -49.5725670, 53.7299080
1: -19.3640194, 43.8358383, -16.1379242, 37.2298279, -56.5938492, 59.9737625
2: -16.6260586, 48.7486916, -13.8656015, 41.4557724, -58.0818329, 62.6142807
3: -18.1543369, 62.5934372, -15.1921024, 53.2694283, -71.4237671, 77.7855377
4: -15.4330111, 57.9811401, -13.0185480, 49.2367287, -64.6697388, 70.9996872

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005529
time: 1.19 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005529
time: 0.93 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -9.6212168, 30.5417042, -40.4928894, 41.1694107
1: -14.1082268, 32.7253380, -13.6478968, 31.6873970, -45.7956238, 46.3732300
2: -12.1443062, 36.5262375, -11.7526855, 35.3688622, -47.5131683, 48.2789154
3: -13.2896729, 46.9455948, -12.8459091, 45.4546242, -58.7442970, 59.7915039
4: -11.4511833, 43.4067154, -11.0942535, 42.0296516, -53.4808273, 54.5009689

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077987, upper bound: 60.1995742
time: 0.94 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113933, upper bound: 60.2036256
time: 0.80 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2181326, upper bound: 60.2181326
time: 1.06 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -9.9511909, 31.5481949, -15.9611063, 48.2339020, -58.1850929, 47.5093002
1: -14.1082268, 32.7253380, -22.2404156, 49.8759995, -63.9842148, 54.9657516
2: -12.1443062, 36.5262375, -19.0615807, 55.5059204, -67.6502151, 55.5878181
3: -13.2896729, 46.9455948, -21.0178299, 71.2318573, -84.5215149, 67.9634247
4: -11.4511833, 43.4067154, -17.6370659, 66.1001282, -77.5513153, 61.0437813

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196737, upper bound: 60.2178055
time: 0.94 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207248, upper bound: 60.2207248
time: 0.89 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -15.9046373, 48.0684547, -7.6289296, 24.6599312, -40.5645676, 55.6973801
1: -22.1601467, 49.7097778, -10.7902393, 25.6240597, -47.7842026, 60.5000153
2: -18.9954109, 55.3179436, -9.2849426, 28.6821880, -47.6775970, 64.6028824
3: -20.9529686, 70.9937897, -10.2551575, 36.8051453, -57.7581139, 81.2489319
4: -17.5727234, 65.8861618, -8.8979664, 34.0427971, -51.6155167, 74.7841263

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2078514, upper bound: 60.2221737
time: 1.11 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2078514, upper bound: 60.2221737
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -15.7180643, 47.6994934, -11.3854551, 35.8409538, -51.5590172, 59.0849495
1: -21.9100761, 49.3196106, -16.1134853, 37.1372223, -59.0472946, 65.4330978
2: -18.7695656, 54.8857689, -13.8432407, 41.3504105, -60.1199760, 68.7290039
3: -20.7376137, 70.4737549, -15.1700706, 53.1259918, -73.8636017, 85.6438217
4: -17.4076424, 65.3452835, -12.9963684, 49.1152878, -66.5229263, 78.3416443

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2115781, upper bound: 60.2228429
time: 0.89 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2160112, upper bound: 60.2160113
time: 1.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.10 seconds
NS_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2185508, upper bound: 60.2258015
NS_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2319440, upper bound: 60.2319440
NS_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2290673, upper bound: 60.2193800
NS_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2319440, upper bound: 60.2319440
NS_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2173488, upper bound: 60.2301730
NS_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2334608, upper bound: 60.2349806
NS_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2180274, upper bound: 60.2301540
NS_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2341204, upper bound: 60.2341204
NS_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2274246, upper bound: 60.2197356
NS_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2295365, upper bound: 60.2226549
NS_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2274246, upper bound: 60.2197356
NS_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2295365, upper bound: 60.2226549
NS_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2107919, upper bound: 60.2249681
NS_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2220471, upper bound: 60.2293598
NS_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2297509, upper bound: 60.2332555
NS_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2260046, upper bound: 60.2332105
NS_B2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2123232
NS_B2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2142216, upper bound: 60.2118209
NS_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2147064, upper bound: 60.2229628
NS_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2142216, upper bound: 60.2118209
NS_B2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2202904, upper bound: 60.1994042
NS_B2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1958138
NS_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005529
NS_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005529
NS_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2113933, upper bound: 60.2036256
NS_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2181326, upper bound: 60.2181326
NS_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2196737, upper bound: 60.2178055
NS_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2207248, upper bound: 60.2207248
NS_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2078514, upper bound: 60.2221737
NS_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2078514, upper bound: 60.2221737
NS_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2115781, upper bound: 60.2228429
NS_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -60.2160112, upper bound: 60.2160113

## BFS NS instance: NS_B2_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -8.4656057, 27.0334034, -5.8457747, 19.0543823, -27.5199833, 32.8791771
1: -12.0761127, 28.1013985, -8.3163891, 19.8533936, -31.9295063, 36.4177856
2: -10.4003716, 31.3413620, -7.1609035, 22.2487202, -32.6490822, 38.5022659
3: -11.3279171, 40.3144455, -7.8484406, 28.5126801, -39.8405876, 48.1628876
4: -9.8721571, 37.2945366, -6.9576654, 26.4493504, -36.3215065, 44.2521896

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132134, upper bound: 60.2165033
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132134, upper bound: 60.2290673
time: 0.75 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -8.3236523, 26.7361107, -9.5343914, 30.2015266, -38.5251694, 36.2705002
1: -11.8846664, 27.7886429, -13.5194168, 31.3581429, -43.2428055, 41.3080521
2: -10.2277231, 30.9924507, -11.6300106, 34.9339485, -45.1616669, 42.6224480
3: -11.1738071, 39.8910904, -12.7315187, 44.8441238, -56.0179291, 52.6226082
4: -9.7274590, 36.8718719, -10.9971132, 41.5406075, -51.2680664, 47.8689842

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290673, upper bound: 60.2193800
time: 1.01 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152254, upper bound: 60.2319440
time: 0.93 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.0000706, 19.5023861, -16.1998177, 48.1688690, -54.1689377, 35.7022018
1: -8.5339622, 20.3191204, -22.4908485, 49.8819237, -58.4158859, 42.8099670
2: -7.3457561, 22.7663746, -19.2778893, 55.4919167, -62.8376732, 42.0442505
3: -8.0549335, 29.1703911, -21.2356243, 71.0937424, -79.1486740, 50.4060135
4: -7.1232290, 27.0667210, -17.7364922, 66.1618881, -73.2851181, 44.8032112

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2299888, upper bound: 60.2024262
time: 0.98 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2287207, upper bound: 60.2170578
time: 0.78 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2144958, upper bound: 60.2022584
time: 0.81 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2311242, upper bound: 60.2161870
time: 0.87 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.7635622, 30.8680763, -15.9730577, 47.6568413, -57.4203949, 46.8411331
1: -13.8402271, 32.0488358, -22.1611881, 49.3422089, -63.1824341, 54.2100220
2: -11.9041538, 35.7016144, -18.9906101, 54.8884239, -66.7925797, 54.6922150
3: -13.0373688, 45.8267136, -20.9605541, 70.3653564, -83.4027252, 66.7872620
4: -11.2463779, 42.4568138, -17.5220947, 65.4203720, -76.6667480, 59.9789085

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185508, upper bound: 60.2258015
time: 0.88 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185508, upper bound: 60.2319440
time: 0.81 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -14.8505383, 44.3763084, -6.8465099, 22.2854996, -37.1360359, 51.2228165
1: -20.5263596, 45.9638596, -9.7379637, 23.1742458, -43.7006073, 55.7018204
2: -17.6609440, 51.1953011, -8.3665676, 25.9154625, -43.5764046, 59.5618668
3: -19.4750385, 65.7007675, -9.2160892, 33.3313942, -52.8064308, 74.9168549
4: -16.3923550, 61.0547218, -8.0679684, 30.8037472, -47.1960907, 69.1226807

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2173488, upper bound: 60.2301730
time: 0.98 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2173488, upper bound: 60.2301730
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -14.7170200, 44.1267586, -11.0595207, 34.7305756, -49.4475937, 55.1862755
1: -20.3464184, 45.7025604, -15.6666641, 36.0120277, -56.3584442, 61.3692245
2: -17.4945450, 50.9034767, -13.4549809, 40.0624428, -57.5569839, 64.3584442
3: -19.3268967, 65.3382645, -14.7530804, 51.5079269, -70.8348236, 80.0913391
4: -16.2762547, 60.6904716, -12.6313782, 47.6448517, -63.9211044, 73.3218536

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2322773, upper bound: 60.2118552
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2176853, upper bound: 60.2130039
time: 0.80 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -16.3075619, 48.6032066, -7.0178251, 22.7734947, -39.0810432, 55.6210251
1: -22.6692619, 50.3288994, -9.9796190, 23.6788063, -46.3480644, 60.3085175
2: -19.4354153, 55.9723625, -8.5698452, 26.4718933, -45.9073067, 64.5422058
3: -21.4130840, 71.7558670, -9.4415588, 34.0342255, -55.4473038, 81.1974258
4: -17.8851261, 66.7344894, -8.2473841, 31.4667988, -49.3519249, 74.9818726

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180274, upper bound: 60.2301540
time: 0.97 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180274, upper bound: 60.2301540
time: 0.97 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -16.0756359, 48.0702438, -11.2176762, 35.1822014, -51.2578316, 59.2879181
1: -22.3320122, 49.7677574, -15.8887405, 36.4774628, -58.8094749, 65.6564941
2: -19.1413860, 55.3447227, -13.6443090, 40.5755043, -59.7168884, 68.9890213
3: -21.1299610, 70.9975967, -14.9605589, 52.1551819, -73.2851410, 85.9581451
4: -17.6637802, 65.9690170, -12.7981911, 48.2544098, -65.9181824, 78.7672119

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2296846, upper bound: 60.2341204
time: 0.80 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2296846, upper bound: 60.2341204
time: 0.94 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -8.5392914, 27.2805252, -36.3778496, 37.6033096
1: -12.8702602, 30.1397648, -12.1856346, 28.3512688, -41.2215271, 42.3254013
2: -11.0743856, 33.6567726, -10.4923859, 31.6131878, -42.6875725, 44.1491585
3: -12.1959352, 43.2404213, -11.4304800, 40.6814766, -52.8774071, 54.6709023
4: -10.4772635, 40.0022278, -9.9563656, 37.6183701, -48.0956306, 49.9585876

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2269287, upper bound: 60.2195355
time: 0.98 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2269287, upper bound: 60.2197356
time: 1.08 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -8.4903440, 27.1595993, -38.0292320, 42.8987389
1: -15.3343544, 35.6384964, -12.1164589, 28.2246857, -43.5590401, 47.7549553
2: -13.1785946, 39.7972412, -10.4313059, 31.4744587, -44.6530533, 50.2285461
3: -14.4879837, 51.1623955, -11.3722286, 40.5064087, -54.9943924, 62.5346222
4: -12.3919563, 47.3149300, -9.9023247, 37.4522400, -49.8441963, 57.2172546

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291584, upper bound: 60.2224549
time: 1.09 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291584, upper bound: 60.2226549
time: 1.01 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -16.3947735, 48.7803421, -57.8776665, 45.4588013
1: -12.8702602, 30.1397648, -22.7656269, 50.5103569, -63.3806114, 52.9053917
2: -11.0743856, 33.6567726, -19.5160103, 56.1796188, -67.2540054, 53.1727753
3: -12.1959352, 43.2404213, -21.4941235, 72.0030670, -84.1989975, 64.7345428
4: -10.4772635, 40.0022278, -17.9616222, 66.9744339, -77.4516830, 57.9638443

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2293353, upper bound: 60.2156704
time: 0.85 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281415, upper bound: 60.2195304
time: 0.91 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -16.3425140, 48.6633453, -59.5329781, 50.7509003
1: -15.3343544, 35.6384964, -22.6888828, 50.3869133, -65.7212601, 58.3273773
2: -13.1785946, 39.7972412, -19.4462547, 56.0446854, -69.2232666, 59.2434883
3: -14.4879837, 51.1623955, -21.4369717, 71.8353195, -86.3233032, 72.5993652
4: -12.3919563, 47.3149300, -17.9083347, 66.8110580, -79.2030029, 65.2232666

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2315818, upper bound: 60.2199738
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2296096, upper bound: 60.2224498
time: 0.95 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -15.9046373, 48.0684547, -5.8457747, 19.0543823, -34.9590187, 53.9142265
1: -22.1601467, 49.7097778, -8.3163891, 19.8533936, -42.0135422, 58.0261688
2: -18.9954109, 55.3179436, -7.1609035, 22.2487202, -41.2441330, 62.4788399
3: -20.9529686, 70.9937897, -7.8484406, 28.5126801, -49.4656487, 78.8422089
4: -17.5727234, 65.8861618, -6.9576654, 26.4493504, -44.0220718, 72.8438263

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2116211, upper bound: 60.2248024
time: 1.08 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2111188, upper bound: 60.2249681
time: 1.44 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -15.7180643, 47.6994934, -9.5343914, 30.2015266, -45.9195900, 57.2338867
1: -21.9100761, 49.3196106, -13.5194168, 31.3581429, -53.2682190, 62.8390198
2: -18.7695656, 54.8857689, -11.6300106, 34.9339485, -53.7035141, 66.5157776
3: -20.7376137, 70.4737549, -12.7315187, 44.8441238, -65.5817413, 83.2052765
4: -17.4076424, 65.3452835, -10.9971132, 41.5406075, -58.9482384, 76.3423920

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156513, upper bound: 60.2291858
time: 0.96 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2199930, upper bound: 60.2240873
time: 1.12 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -16.1192989, 48.7391510, -15.6896954, 46.6440086, -62.7633057, 64.4288406
1: -22.4673023, 50.3988075, -21.7830467, 48.3114319, -70.7787323, 72.1818542
2: -19.2585125, 56.0775299, -18.6801777, 53.7499046, -73.0084152, 74.7577057
3: -21.2393341, 71.9838791, -20.5744972, 68.8453522, -90.0846863, 92.5583801
4: -17.8179779, 66.7789383, -17.1931114, 64.1020660, -81.9200439, 83.9720383

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215413, upper bound: 60.2176047
time: 1.01 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2244024, upper bound: 60.2299042
time: 0.86 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -16.0599804, 48.5664787, -17.4189968, 50.9589958, -67.0189743, 65.9854736
1: -22.3833942, 50.2205276, -24.0174179, 52.7902489, -75.1736450, 74.2379303
2: -19.1867943, 55.8817711, -20.6385117, 58.8075638, -77.9943466, 76.5202789
3: -21.1627502, 71.7335663, -22.7403164, 75.1969604, -96.3597031, 94.4738846
4: -17.7557144, 66.5460663, -18.9181595, 70.1685486, -87.9242630, 85.4642181

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180427, upper bound: 60.2160431
time: 0.91 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2230702, upper bound: 60.2293952
time: 0.92 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -6.6016684, 21.5060062, -7.8912673, 25.4425735, -32.0442352, 29.3972721
1: -9.3999586, 22.3729630, -11.1526108, 26.4303532, -35.8303070, 33.5255699
2: -8.0780697, 25.0461235, -9.5916300, 29.5862541, -37.6643219, 34.6377525
3: -8.9064245, 32.1721535, -10.6004381, 37.9559097, -46.8623352, 42.7725906
4: -7.8093925, 29.7681084, -9.1747046, 35.1177864, -42.9271774, 38.9428139

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2074786
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2118209
time: 1.09 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -7.6358347, 24.1759071, -7.8488779, 25.3174496, -32.9532852, 32.0247841
1: -10.7233191, 25.1112423, -11.0934715, 26.3006096, -37.0239296, 36.2047119
2: -9.2092772, 28.1248932, -9.5414448, 29.4444542, -38.6537323, 37.6663361
3: -10.1858873, 36.0929527, -10.5446796, 37.7743988, -47.9602852, 46.6376305
4: -8.8083553, 33.4915848, -9.1302462, 34.9497337, -43.7580872, 42.6218262

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2074786
time: 1.13 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2118209
time: 0.89 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -10.8226824, 33.9893799, -7.8912673, 25.4425735, -36.2652550, 41.8806458
1: -15.3359919, 35.2548904, -11.1526108, 26.4303532, -41.7663383, 46.4074974
2: -13.1706419, 39.2478790, -9.5916300, 29.5862541, -42.7568970, 48.8395081
3: -14.4467955, 50.4317055, -10.6004381, 37.9559097, -52.4027061, 61.0321426
4: -12.3785849, 46.6736145, -9.1747046, 35.1177864, -47.4963684, 55.8483162

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2092813
time: 0.98 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2029703, upper bound: 60.2229628
time: 0.92 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -11.6891575, 36.3419762, -7.8488779, 25.3174496, -37.0066071, 44.1908531
1: -16.4537983, 37.6537743, -11.0934715, 26.3006096, -42.7544098, 48.7472420
2: -14.1387310, 41.9565811, -9.5414448, 29.4444542, -43.5831757, 51.4980240
3: -15.5197172, 53.9093857, -10.5446796, 37.7743988, -53.2941170, 64.4540634
4: -13.2439423, 49.9412079, -9.1302462, 34.9497337, -48.1936760, 59.0714531

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2029703, upper bound: 60.2110524
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2029703, upper bound: 60.2251931
time: 1.02 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -9.4499655, 30.1438122, -11.5309305, 36.2892380, -45.7392006, 41.6747360
1: -13.4802599, 31.2940292, -16.3129578, 37.6009598, -51.0812149, 47.6069832
2: -11.5837431, 34.8744049, -14.0159235, 41.8694153, -53.4531555, 48.8903236
3: -12.6736584, 44.9068756, -15.3604355, 53.7925797, -66.4662399, 60.2673111
4: -10.9627438, 41.4599915, -13.1525908, 49.7282600, -60.6910019, 54.6125755

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202904, upper bound: 60.1994042
time: 0.96 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202904, upper bound: 60.1994042
time: 0.84 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -9.4896736, 30.2163372, -11.5309305, 36.2892380, -45.7789001, 41.7472572
1: -13.5354614, 31.3710442, -16.3129578, 37.6009598, -51.1364212, 47.6839981
2: -11.6318226, 34.9533844, -14.0159235, 41.8694153, -53.5012360, 48.9693031
3: -12.7202435, 45.0115929, -15.3604355, 53.7925797, -66.5128250, 60.3720245
4: -11.0013351, 41.5614967, -13.1525908, 49.7282600, -60.7295914, 54.7140770

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1958138
time: 0.92 seconds

## Relational analysis of NS_B2_B2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1958138
time: 0.93 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -13.6414785, 42.3319702, -10.9193020, 34.4716988, -48.1131783, 53.2512703
1: -19.3640194, 43.8358383, -15.4629669, 35.7357826, -55.0998001, 59.2988052
2: -16.6260586, 48.7486916, -13.2871695, 39.8115616, -56.4376221, 62.0358620
3: -18.1543369, 62.5934372, -14.5526495, 51.1244621, -69.2788010, 77.1460876
4: -15.4330111, 57.9811401, -12.4893827, 47.2725906, -62.7056007, 70.4705200

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073385, upper bound: 60.1993244
time: 1.02 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2012072, upper bound: 60.1978744
time: 0.88 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -13.6414785, 42.3319702, -14.8790970, 46.2554855, -59.8969650, 57.2110596
1: -19.3640194, 43.8358383, -21.0219193, 47.8620033, -67.2260132, 64.8577576
2: -16.6260586, 48.7486916, -18.0474167, 53.2342529, -69.8603134, 66.7961121
3: -18.1543369, 62.5934372, -19.7871208, 68.2334290, -86.3877563, 82.3805542
4: -15.4330111, 57.9811401, -16.7340488, 63.2242813, -78.6572952, 74.7151871

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005529
time: 0.87 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2053892, upper bound: 60.1933258
time: 1.09 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.2605543, 26.1002769, -9.6212168, 30.5417042, -38.8022575, 35.7214928
1: -11.6701870, 27.1455231, -13.6478968, 31.6873970, -43.3575821, 40.7934189
2: -10.0508842, 30.4125671, -11.7526855, 35.3688622, -45.4197426, 42.1652412
3: -10.9955826, 38.9221077, -12.8459091, 45.4546242, -56.4502068, 51.7680168
4: -9.5514736, 36.0904274, -11.0942535, 42.0296516, -51.5811234, 47.1846809

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1968863, upper bound: 60.1968863
time: 0.90 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1968863, upper bound: 60.2036256
time: 0.95 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.7334709, 30.9533539, -9.6212168, 30.5417042, -40.2751617, 40.5745697
1: -13.8223515, 32.1135483, -13.6478968, 31.6873970, -45.5097466, 45.7614441
2: -11.9020767, 35.8392563, -11.7526855, 35.3688622, -47.2709389, 47.5919342
3: -13.0118752, 46.0697556, -12.8459091, 45.4546242, -58.4664993, 58.9156647
4: -11.2337151, 42.5850677, -11.0942535, 42.0296516, -53.2633667, 53.6793175

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2036256, upper bound: 60.2113933
time: 0.99 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2036256, upper bound: 60.2181326
time: 0.79 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -15.9611063, 48.2339020, -57.3312263, 45.0251350
1: -12.8702602, 30.1397648, -22.2404156, 49.8759995, -62.7462540, 52.3801804
2: -11.0743856, 33.6567726, -19.0615807, 55.5059204, -66.5803070, 52.7183533
3: -12.1959352, 43.2404213, -21.0178299, 71.2318573, -83.4277878, 64.2582550
4: -10.4772635, 40.0022278, -17.6370659, 66.1001282, -76.5773773, 57.6392937

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2271675, upper bound: 60.2069831
time: 1.02 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2257592, upper bound: 60.2174455
time: 0.96 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -15.9093161, 48.1180305, -58.9876633, 50.3177071
1: -15.3343544, 35.6384964, -22.1674767, 49.7544174, -65.0887680, 57.8059731
2: -13.1785946, 39.7972412, -18.9941292, 55.3722420, -68.5508270, 58.7913666
3: -14.4879837, 51.1623955, -20.9614658, 71.0657120, -85.5536957, 72.1238632
4: -12.3919563, 47.3149300, -17.5844803, 65.9381027, -78.3300476, 64.8994064

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2296625, upper bound: 60.2099024
time: 0.96 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2267164, upper bound: 60.2203648
time: 0.99 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -15.9046373, 48.0684547, -6.7173672, 21.7505741, -37.6552124, 54.7858162
1: -22.1601467, 49.7097778, -9.4661121, 22.6145306, -44.7746735, 59.1758881
2: -18.9954109, 55.3179436, -8.1495190, 25.3800755, -44.3754730, 63.4674568
3: -20.9529686, 70.9937897, -9.0071821, 32.5074387, -53.4604073, 80.0009689
4: -17.5727234, 65.8861618, -7.8611236, 30.1524315, -47.7251511, 73.7472839

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2029435, upper bound: 60.2217984
time: 0.93 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073766, upper bound: 60.2168155
time: 0.90 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -15.9046373, 48.0684547, -11.9908962, 36.5895920, -52.4942284, 60.0593491
1: -22.1601467, 49.7097778, -16.6667862, 37.8833618, -60.0435104, 66.3765564
2: -18.9954109, 55.3179436, -14.2192478, 42.3016624, -61.2970619, 69.5371933
3: -20.9529686, 70.9937897, -15.8255281, 54.0901794, -75.0431213, 86.8193207
4: -17.5727234, 65.8861618, -13.2623749, 50.4039803, -67.9766998, 79.1485367

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075152, upper bound: 60.2105480
time: 1.24 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075152, upper bound: 60.2105480
time: 0.95 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.2701521, 43.4083481, -11.2214355, 35.3575287, -49.6276817, 54.6297836
1: -19.7633266, 44.9080849, -15.8829308, 36.6401596, -56.4034882, 60.7910156
2: -17.0046043, 50.0661469, -13.6480532, 40.8009529, -57.8055573, 63.7141991
3: -18.8302383, 64.3283615, -14.9547825, 52.4313545, -71.2615738, 79.2831345
4: -15.8524036, 59.6322556, -12.8253012, 48.4659767, -64.3183746, 72.4575577

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2115781, upper bound: 60.2115781
time: 0.92 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2115781, upper bound: 60.2160113
time: 0.95 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.4533587, 46.9513283, -11.3854551, 35.8409538, -51.2943115, 58.3367844
1: -21.5374165, 48.5503235, -16.1134853, 37.1372223, -58.6746368, 64.6638107
2: -18.4542274, 54.0260010, -13.8432407, 41.3504105, -59.8046265, 67.8692398
3: -20.3998470, 69.3874435, -15.1700706, 53.1259918, -73.5258255, 84.5574951
4: -17.1273994, 64.3239212, -12.9963684, 49.1152878, -66.2426910, 77.3202820

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090898, upper bound: 60.2093540
time: 0.94 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090898, upper bound: 60.2160113
time: 0.89 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.75 seconds
NS_B2_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2132134, upper bound: 60.2165033
NS_B2_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2132134, upper bound: 60.2290673
NS_B2_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2290673, upper bound: 60.2193800
NS_B2_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2152254, upper bound: 60.2319440
NS_B2_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2144958, upper bound: 60.2022584
NS_B2_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2311242, upper bound: 60.2161870
NS_B2_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2185508, upper bound: 60.2258015
NS_B2_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2185508, upper bound: 60.2319440
NS_B2_B1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2173488, upper bound: 60.2301730
NS_B2_B1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2173488, upper bound: 60.2301730
NS_B2_B1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2322773, upper bound: 60.2118552
NS_B2_B1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2176853, upper bound: 60.2130039
NS_B2_B1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2180274, upper bound: 60.2301540
NS_B2_B1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2180274, upper bound: 60.2301540
NS_B2_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2296846, upper bound: 60.2341204
NS_B2_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2296846, upper bound: 60.2341204
NS_B2_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2269287, upper bound: 60.2195355
NS_B2_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2269287, upper bound: 60.2197356
NS_B2_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2291584, upper bound: 60.2224549
NS_B2_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2291584, upper bound: 60.2226549
NS_B2_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2293353, upper bound: 60.2156704
NS_B2_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2281415, upper bound: 60.2195304
NS_B2_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2315818, upper bound: 60.2199738
NS_B2_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2296096, upper bound: 60.2224498
NS_B2_B1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2116211, upper bound: 60.2248024
NS_B2_B1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2111188, upper bound: 60.2249681
NS_B2_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2156513, upper bound: 60.2291858
NS_B2_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2199930, upper bound: 60.2240873
NS_B2_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2215413, upper bound: 60.2176047
NS_B2_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2244024, upper bound: 60.2299042
NS_B2_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2180427, upper bound: 60.2160431
NS_B2_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2230702, upper bound: 60.2293952
NS_B2_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2074786
NS_B2_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2118209
NS_B2_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2074786
NS_B2_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2118209
NS_B2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2017944, upper bound: 60.2092813
NS_B2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2029703, upper bound: 60.2229628
NS_B2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2029703, upper bound: 60.2110524
NS_B2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2029703, upper bound: 60.2251931
NS_B2_B2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2202904, upper bound: 60.1994042
NS_B2_B2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2202904, upper bound: 60.1994042
NS_B2_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1958138
NS_B2_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2218416, upper bound: 60.1958138
NS_B2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2073385, upper bound: 60.1993244
NS_B2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2012072, upper bound: 60.1978744
NS_B2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2073451, upper bound: 60.2005529
NS_B2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2053892, upper bound: 60.1933258
NS_B2_B2_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.1968863, upper bound: 60.1968863
NS_B2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.1968863, upper bound: 60.2036256
NS_B2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2036256, upper bound: 60.2113933
NS_B2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2036256, upper bound: 60.2181326
NS_B2_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2271675, upper bound: 60.2069831
NS_B2_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2257592, upper bound: 60.2174455
NS_B2_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2296625, upper bound: 60.2099024
NS_B2_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2267164, upper bound: 60.2203648
NS_B2_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2029435, upper bound: 60.2217984
NS_B2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2073766, upper bound: 60.2168155
NS_B2_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2075152, upper bound: 60.2105480
NS_B2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2075152, upper bound: 60.2105480
NS_B2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2115781, upper bound: 60.2115781
NS_B2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2115781, upper bound: 60.2160113
NS_B2_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2090898, upper bound: 60.2093540
NS_B2_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.75
Output dim: 4, lower bound: -60.2090898, upper bound: 60.2160113

## BFS NS instance: NS_B2_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -6.0000706, 19.5023861, -5.8457747, 19.0543823, -25.0544491, 25.3481598
1: -8.5339622, 20.3191204, -8.3163891, 19.8533936, -28.3873558, 28.6355095
2: -7.3457561, 22.7663746, -7.1609035, 22.2487202, -29.5944767, 29.9272785
3: -8.0549335, 29.1703911, -7.8484406, 28.5126801, -36.5676079, 37.0188332
4: -7.1232290, 27.0667210, -6.9576654, 26.4493504, -33.5725784, 34.0243797

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110082, upper bound: 60.2145155
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164345, upper bound: 60.2165033
time: 1.08 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -9.7635622, 30.8680763, -5.8457747, 19.0543823, -28.8179436, 36.7138519
1: -13.8402271, 32.0488358, -8.3163891, 19.8533936, -33.6936188, 40.3652267
2: -11.9041538, 35.7016144, -7.1609035, 22.2487202, -34.1528740, 42.8625145
3: -13.0373688, 45.8267136, -7.8484406, 28.5126801, -41.5500488, 53.6751556
4: -11.2463779, 42.4568138, -6.9576654, 26.4493504, -37.6957283, 49.4144707

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2119679, upper bound: 60.2287087
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B1_A2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161446, upper bound: 60.2269727
time: 0.83 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -6.0000706, 19.5023861, -9.5343914, 30.2015266, -36.2015953, 29.0367756
1: -8.5339622, 20.3191204, -13.5194168, 31.3581429, -39.8921051, 33.8385391
2: -7.3457561, 22.7663746, -11.6300106, 34.9339485, -42.2797050, 34.3963699
3: -8.0549335, 29.1703911, -12.7315187, 44.8441238, -52.8990517, 41.9019012
4: -7.1232290, 27.0667210, -10.9971132, 41.5406075, -48.6638374, 38.0638351

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134093, upper bound: 60.2192391
time: 0.90 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2145155, upper bound: 60.2192455
time: 0.88 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2102229, upper bound: 60.2193800
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -9.7635622, 30.8680763, -9.5343914, 30.2015266, -39.9650764, 40.4024658
1: -13.8402271, 32.0488358, -13.5194168, 31.3581429, -45.1983719, 45.5682526
2: -11.9041538, 35.7016144, -11.6300106, 34.9339485, -46.8381042, 47.3316193
3: -13.0373688, 45.8267136, -12.7315187, 44.8441238, -57.8814888, 58.5582237
4: -11.2463779, 42.4568138, -10.9971132, 41.5406075, -52.7869873, 53.4539261

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2138393, upper bound: 60.2269975
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2128114, upper bound: 60.2269435
time: 0.95 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -4.3818970, 14.2773447, -16.1998177, 48.1688690, -52.5507584, 30.4771614
1: -6.2034554, 14.9397087, -22.4908485, 49.8819237, -56.0853691, 37.4305496
2: -5.3013792, 16.8548222, -19.2778893, 55.4919167, -60.7932968, 36.1327095
3: -5.8983307, 21.4165840, -21.2356243, 71.0937424, -76.9920578, 42.6522064
4: -5.2637382, 19.9639473, -17.7364922, 66.1618881, -71.4256287, 37.7004395

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2092522, upper bound: 60.1998574
time: 0.87 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113678, upper bound: 60.2000754
time: 0.91 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -5.8169990, 19.0006428, -16.1998177, 48.1688690, -53.9858665, 35.2004623
1: -8.2833433, 19.7987843, -22.4908485, 49.8819237, -58.1652641, 42.2896347
2: -7.1355338, 22.1896782, -19.2778893, 55.4919167, -62.6274490, 41.4675598
3: -7.8154688, 28.4313526, -21.2356243, 71.0937424, -78.9092026, 49.6669769
4: -6.9357119, 26.3707867, -17.7364922, 66.1618881, -73.0975952, 44.1072731

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2278221, upper bound: 60.2139316
time: 0.97 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2296765, upper bound: 60.2139316
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.7635622, 30.8680763, -13.9419765, 41.3171806, -51.0807343, 44.8100510
1: -13.8402271, 32.0488358, -19.3152542, 42.8444366, -56.6846619, 51.3640900
2: -11.9041538, 35.7016144, -16.5293350, 47.7828712, -59.6870270, 52.2309418
3: -13.0373688, 45.8267136, -18.2614937, 60.7640800, -73.8014450, 64.0882034
4: -11.2463779, 42.4568138, -15.1969624, 56.9263229, -68.1726990, 57.6537743

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161821, upper bound: 60.2182054
time: 1.03 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167365, upper bound: 60.2226852
time: 0.92 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162293, upper bound: 60.2228135
time: 0.89 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.7635622, 30.8680763, -17.6799107, 52.4844742, -62.2480316, 48.5479813
1: -13.8402271, 32.0488358, -24.5549698, 54.3099556, -68.1501846, 56.6038055
2: -11.9041538, 35.7016144, -20.9728394, 60.4024658, -72.3066177, 56.6744537
3: -13.0373688, 45.8267136, -23.1829853, 77.1572037, -90.1945724, 69.0096817
4: -11.2463779, 42.4568138, -19.2195110, 71.9479599, -83.1943359, 61.6763229

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167365, upper bound: 60.2295391
time: 0.79 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162293, upper bound: 60.2287246
time: 1.05 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -14.8505383, 44.3763084, -5.6798420, 18.5699215, -33.4204597, 50.0561447
1: -20.5263596, 45.9638596, -8.0816326, 19.3503017, -39.8766556, 54.0454941
2: -17.6609440, 51.1953011, -6.9639864, 21.6942596, -39.3552017, 58.1592827
3: -19.4750385, 65.7007675, -7.6264119, 27.8118057, -47.2868423, 73.3271790
4: -16.3923550, 61.0547218, -6.7831469, 25.7893600, -42.1817169, 67.8378677

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2165158, upper bound: 60.2295167
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147651, upper bound: 60.2260379
time: 1.13 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2078290, upper bound: 60.2284745
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -14.8505383, 44.3763084, -12.6307526, 37.6716728, -52.5222092, 57.0070610
1: -20.5263596, 45.9638596, -17.4894466, 39.0664825, -59.5928383, 63.4533081
2: -17.6609440, 51.1953011, -14.9071369, 43.6241608, -61.2850914, 66.1024399
3: -19.4750385, 65.7007675, -16.5991821, 55.5454826, -75.0205078, 82.2999496
4: -16.3923550, 61.0547218, -13.7728176, 51.9786148, -68.3709717, 74.8275375

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2049221, upper bound: 60.2183898
time: 0.89 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2139159, upper bound: 60.2270768
time: 0.81 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14.2588949, 42.6475601, -11.0595207, 34.7305756, -48.9894676, 53.7070770
1: -19.7009792, 44.1850319, -15.6666641, 36.0120277, -55.7130013, 59.8516960
2: -16.9404945, 49.2349205, -13.4549809, 40.0624428, -57.0029373, 62.6898918
3: -18.7140465, 63.1503220, -14.7530804, 51.5079269, -70.2219696, 77.9033890
4: -15.7540874, 58.7054596, -12.6313782, 47.6448517, -63.3989410, 71.3368073

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2299465, upper bound: 60.2087180
time: 0.98 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2230991, upper bound: 60.2094621
time: 1.25 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.9744415, 50.7706947, -10.9559984, 34.4493637, -51.4238052, 61.7266884
1: -23.4558029, 52.5290833, -15.5297003, 35.7234879, -59.1792870, 68.0587769
2: -20.0809212, 58.4921951, -13.3389406, 39.7404900, -59.8214035, 71.8311234
3: -22.1786594, 74.9103699, -14.6224146, 51.0983772, -73.2770386, 89.5327835
4: -18.5901947, 69.6463470, -12.5272617, 47.2599602, -65.8501587, 82.1736069

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174673, upper bound: 60.2130039
time: 0.69 seconds

## Relational analysis of NS_B2_B1_A1_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174673, upper bound: 60.2130039
time: 1.03 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -16.3075619, 48.6032066, -5.8457747, 19.0543823, -35.3619385, 54.4489822
1: -22.6692619, 50.3288994, -8.3163891, 19.8533936, -42.5226555, 58.6452866
2: -19.4354153, 55.9723625, -7.1609035, 22.2487202, -41.6841354, 63.1332664
3: -21.4130840, 71.7558670, -7.8484406, 28.5126801, -49.9257584, 79.6042938
4: -17.8851261, 66.7344894, -6.9576654, 26.4493504, -44.3344765, 73.6921539

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1954448, upper bound: 60.2284310
time: 1.45 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2175722, upper bound: 60.2294971
time: 1.08 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2176913, upper bound: 60.2231574
time: 1.05 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -16.3075619, 48.6032066, -12.8284712, 38.2544174, -54.5619698, 61.4316750
1: -22.6692619, 50.3288994, -17.7783699, 39.6664696, -62.3357315, 68.1072693
2: -19.4354153, 55.9723625, -15.1493225, 44.2929230, -63.7283287, 71.1216888
3: -21.4130840, 71.7558670, -16.8606548, 56.3715591, -77.7846451, 88.6165237
4: -17.8851261, 66.7344894, -13.9779129, 52.7644882, -70.6496124, 80.7124023

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140653, upper bound: 60.2140653
time: 0.81 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140653, upper bound: 60.2140653
time: 0.91 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -16.0756359, 48.0702438, -9.5343914, 30.2015266, -46.2771530, 57.6046333
1: -22.3320122, 49.7677574, -13.5194168, 31.3581429, -53.6901550, 63.2871628
2: -19.1413860, 55.3447227, -11.6300106, 34.9339485, -54.0753326, 66.9747314
3: -21.1299610, 70.9975967, -12.7315187, 44.8441238, -65.9740829, 83.7291183
4: -17.6637802, 65.9690170, -10.9971132, 41.5406075, -59.2043800, 76.9661331

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140653, upper bound: 60.2180275
time: 1.02 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140653, upper bound: 60.2140653
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -16.0756359, 48.0702438, -17.6799107, 52.4844742, -68.5601044, 65.7501450
1: -22.3320122, 49.7677574, -24.5549698, 54.3099556, -76.6419678, 74.3227234
2: -19.1413860, 55.3447227, -20.9728394, 60.4024658, -79.5438461, 76.3175430
3: -21.1299610, 70.9975967, -23.1829853, 77.1572037, -98.2871628, 94.1805649
4: -17.6637802, 65.9690170, -19.2195110, 71.9479599, -89.6117325, 85.1885300

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B2_B1

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2278727, upper bound: 60.2336101
time: 0.89 seconds

## Relational analysis of NS_B2_B1_A1_A2_A2_B2_B2_B2

### Relational analysis result of NS_B2_B1_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193736, upper bound: 60.2321362
time: 1.22 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -7.6743560, 24.6581879, -33.7555122, 36.7383728
1: -12.8702602, 30.1397648, -10.9328156, 25.6382370, -38.5084953, 41.0725784
2: -11.0743856, 33.6567726, -9.4233618, 28.5977440, -39.6721306, 43.0801315
3: -12.1959352, 43.2404213, -10.2980270, 36.7600975, -48.9560280, 53.5384483
4: -10.4772635, 40.0022278, -8.9728336, 34.0330658, -44.5103264, 48.9750595

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2159060, upper bound: 60.2035215
time: 0.77 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241591, upper bound: 60.2153403
time: 0.92 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -9.9168406, 31.2821922, -40.3795128, 38.9808693
1: -12.8702602, 30.1397648, -13.9611149, 32.4701347, -45.3403931, 44.1008797
2: -11.0743856, 33.6567726, -11.9938822, 36.2576675, -47.3320541, 45.6506538
3: -12.1959352, 43.2404213, -13.2197113, 46.5242653, -58.7201996, 56.4601326
4: -10.4772635, 40.0022278, -11.3215113, 43.1129227, -53.5901833, 51.3237381

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2159060, upper bound: 60.2037042
time: 0.87 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241591, upper bound: 60.2155404
time: 0.98 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -7.6743560, 24.6581879, -35.5278206, 42.0827484
1: -15.3343544, 35.6384964, -10.9328156, 25.6382370, -40.9725876, 46.5713120
2: -13.1785946, 39.7972412, -9.4233618, 28.5977440, -41.7763367, 49.2205963
3: -14.4879837, 51.1623955, -10.2980270, 36.7600975, -51.2480736, 61.4604187
4: -12.3919563, 47.3149300, -8.9728336, 34.0330658, -46.4250221, 56.2877655

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2275119, upper bound: 60.2202793
time: 0.90 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2235941, upper bound: 60.2203179
time: 0.80 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -9.9168406, 31.2821922, -42.1518250, 44.3252335
1: -15.3343544, 35.6384964, -13.9611149, 32.4701347, -47.8044891, 49.5996094
2: -13.1785946, 39.7972412, -11.9938822, 36.2576675, -49.4362640, 51.7911224
3: -14.4879837, 51.1623955, -13.2197113, 46.5242653, -61.0122452, 64.3821106
4: -12.3919563, 47.3149300, -11.3215113, 43.1129227, -55.5048790, 58.6364403

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1867325, upper bound: 60.1818879
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2269801, upper bound: 60.2197076
time: 0.96 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.9127254, 28.5410595, -14.6994267, 43.8830948, -52.7958221, 43.2404785
1: -12.6067314, 29.6007442, -20.2262688, 45.4584618, -58.0651894, 49.8270111
2: -10.8521481, 33.0609627, -17.4245892, 50.6726418, -61.5247803, 50.4855461
3: -11.9536343, 42.4839859, -19.2903652, 64.9980774, -76.9517136, 61.7743530
4: -10.2816601, 39.2907829, -16.1747570, 60.4101944, -70.6918564, 55.4655304

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2267470, upper bound: 60.2085274
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2271563, upper bound: 60.2089507
time: 0.98 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -16.1591396, 48.1012154, -57.1985359, 45.2231636
1: -12.8702602, 30.1397648, -22.4312439, 49.8136253, -62.6838799, 52.5710068
2: -11.0743856, 33.6567726, -19.2339249, 55.4063568, -66.4807434, 52.8906975
3: -12.1959352, 43.2404213, -21.1944542, 71.0239105, -83.2198410, 64.4348755
4: -10.4772635, 40.0022278, -17.7075386, 66.0645752, -76.5418243, 57.7097664

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2085796, upper bound: 60.1981318
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213731, upper bound: 60.2107541
time: 1.09 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2279609, upper bound: 60.2195304
time: 1.19 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.6968594, 33.9088364, -14.6554956, 43.7864532, -54.4833069, 48.5643311
1: -15.0916224, 35.1240158, -20.1648464, 45.3563881, -60.4479980, 55.2888641
2: -12.9737864, 39.2294922, -17.3666019, 50.5601120, -63.5338898, 56.5960922
3: -14.2583523, 50.4437637, -19.2440548, 64.8593979, -79.1177521, 69.6878052
4: -12.2118120, 46.6417999, -16.1307278, 60.2763329, -72.4881439, 62.7725296

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2303297, upper bound: 60.2199738
time: 0.93 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2311028, upper bound: 60.2187539
time: 0.93 seconds

## BFS NS instance: NS_B2_B1_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -16.1061096, 47.9789009, -58.8485336, 50.5144997
1: -15.3343544, 35.6384964, -22.3534279, 49.6858292, -65.0201797, 57.9919243
2: -13.1785946, 39.7972412, -19.1632118, 55.2646027, -68.4431763, 58.9604530
3: -14.4879837, 51.1623955, -21.1358433, 70.8514862, -85.3394699, 72.2982407
4: -12.3919563, 47.3149300, -17.6531620, 65.8964081, -78.2883530, 64.9680939

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2293359, upper bound: 60.2224498
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_B1_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290171, upper bound: 60.2224498
time: 0.94 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -15.9046373, 48.0684547, -5.2172346, 17.1237125, -33.0283508, 53.2856865
1: -22.1601467, 49.7097778, -7.4405303, 17.8501015, -40.0102463, 57.1503067
2: -18.9954109, 55.3179436, -6.4116488, 20.0427399, -39.0381508, 61.7295761
3: -20.9529686, 70.9937897, -7.0220485, 25.6635628, -46.6165314, 78.0158310
4: -17.5727234, 65.8861618, -6.2815475, 23.8232307, -41.3959541, 72.1677094

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2067132, upper bound: 60.2242572
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2108530, upper bound: 60.2205512
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2105125, upper bound: 60.2213320
time: 0.90 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -15.8446236, 47.8950615, -6.3904877, 20.1030827, -35.9477005, 54.2855492
1: -22.0751972, 49.5307121, -8.9238720, 20.9166241, -42.9918213, 58.4545822
2: -18.9229088, 55.1213646, -7.6799192, 23.4965019, -42.4194107, 62.8012733
3: -20.8759727, 70.7425308, -8.4619112, 30.0532818, -50.9292526, 79.2044449
4: -17.5098877, 65.6522827, -7.3969960, 27.9947701, -45.5046577, 73.0492783

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2062109, upper bound: 60.2244252
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2106440, upper bound: 60.2203324
time: 1.06 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -14.2701521, 43.4083481, -9.3446083, 29.6694221, -43.9395752, 52.7529526
1: -19.7633266, 44.9080849, -13.2514830, 30.8084164, -50.5717430, 58.1595688
2: -17.0046043, 50.0661469, -11.4020271, 34.3269501, -51.3315506, 61.4681702
3: -18.8302383, 64.3283615, -12.4837914, 44.0824738, -62.9127121, 76.8121490
4: -15.8524036, 59.6322556, -10.7967253, 40.8200798, -56.6724739, 70.4289780

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148978, upper bound: 60.2253627
time: 1.14 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148978, upper bound: 60.2284941
time: 0.93 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -15.4533587, 46.9513283, -9.5343914, 30.2015266, -45.6548843, 56.4857178
1: -21.5374165, 48.5503235, -13.5194168, 31.3581429, -52.8955536, 62.0697250
2: -18.4542274, 54.0260010, -11.6300106, 34.9339485, -53.3881683, 65.6560135
3: -20.3998470, 69.3874435, -12.7315187, 44.8441238, -65.2439728, 82.1189575
4: -17.1273994, 64.3239212, -10.9971132, 41.5406075, -58.6680031, 75.3210373

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197557, upper bound: 60.2240873
time: 0.95 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131583, upper bound: 60.2218273
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -13.3044767, 40.1446075, -15.5016356, 46.0480499, -59.3525276, 55.6462402
1: -18.4690800, 41.5440445, -21.5175514, 47.7000732, -66.1691437, 63.0615959
2: -15.8064594, 46.3984032, -18.4497986, 53.0925484, -68.8989792, 64.8482056
3: -17.5225220, 59.2194366, -20.3255100, 67.9601822, -85.4826965, 79.5449448
4: -14.6301203, 55.2837143, -16.9745731, 63.3114967, -77.9416122, 72.2582855

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2211348, upper bound: 60.2174500
time: 1.05 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2199839, upper bound: 60.2174127
time: 0.89 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -16.8450050, 51.1080399, -15.2932825, 45.5783501, -62.4233551, 66.4013138
1: -23.4218330, 52.8250275, -21.2102737, 47.2085838, -70.6303864, 74.0353012
2: -20.0210838, 58.7532997, -18.1823978, 52.5220108, -72.5430908, 76.9356918
3: -22.2593422, 75.3798065, -20.0720062, 67.2990952, -89.5584183, 95.4518127
4: -18.5738354, 69.9567947, -16.7776909, 62.6203003, -81.1941376, 86.7344818

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152403, upper bound: 60.2292721
time: 0.82 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196480, upper bound: 60.2240699
time: 1.01 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -13.2425547, 39.9563522, -17.2297077, 50.3573418, -63.5998955, 57.1860504
1: -18.3801975, 41.3501358, -23.7487164, 52.1738739, -70.5540619, 65.0988541
2: -15.7314644, 46.1865578, -20.4056740, 58.1444702, -73.8759308, 66.5922318
3: -17.4421902, 58.9499397, -22.4898949, 74.3022537, -91.7444382, 81.4398270
4: -14.5643787, 55.0331383, -18.7060394, 69.3738098, -83.9381866, 73.7391739

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104558, upper bound: 60.2133048
time: 0.84 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104558, upper bound: 60.2160431
time: 1.05 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -16.7918472, 50.9487610, -17.0175629, 49.8874397, -66.6792755, 67.9663239
1: -23.3461895, 52.6602211, -23.4445667, 51.6758842, -75.0220718, 76.1047897
2: -19.9585514, 58.5739212, -20.1439762, 57.5275650, -77.4861145, 78.7178879
3: -22.1896610, 75.1499557, -22.2298756, 73.6569519, -95.8466110, 97.3798294
4: -18.5179958, 69.7442703, -18.4854412, 68.6698685, -87.1878433, 88.2297134

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1987546, upper bound: 60.2214192
time: 0.81 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2003581, upper bound: 60.2073451
time: 1.11 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6.6016684, 21.5060062, -7.1544199, 23.2220268, -29.8236904, 28.6604271
1: -9.3999586, 22.3729630, -10.1142340, 24.1331444, -33.5331039, 32.4871979
2: -8.0780697, 25.0461235, -8.7066746, 27.0448685, -35.1229401, 33.7527962
3: -8.9064245, 32.1721535, -9.6192160, 34.7037201, -43.6101456, 41.7913704
4: -7.8093925, 29.7681084, -8.3802814, 32.1033325, -39.9127235, 38.1483917

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1886302, upper bound: 60.1946096
time: 0.92 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1752758, upper bound: 60.1749117
time: 0.77 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6.6016684, 21.5060062, -8.4977674, 26.8933430, -33.4950066, 30.0037670
1: -9.3999586, 22.3729630, -11.8817339, 27.8938255, -37.2937851, 34.2546844
2: -8.0780697, 25.0461235, -10.1987410, 31.2466660, -39.3247375, 35.2448654
3: -8.9064245, 32.1721535, -11.3400717, 40.0894585, -48.9958839, 43.5122223
4: -7.8093925, 29.7681084, -9.7117977, 37.1281395, -44.9375305, 39.4799042

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1975069, upper bound: 60.2086083
time: 0.93 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1886302, upper bound: 60.1946096
time: 0.95 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1752758, upper bound: 60.1749117
time: 0.88 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6358347, 24.1759071, -7.1544199, 23.2220268, -30.8578587, 31.3303242
1: -10.7233191, 25.1112423, -10.1142340, 24.1331444, -34.8564606, 35.2254753
2: -9.2092772, 28.1248932, -8.7066746, 27.0448685, -36.2541466, 36.8315659
3: -10.1858873, 36.0929527, -9.6192160, 34.7037201, -44.8896065, 45.7121696
4: -8.8083553, 33.4915848, -8.3802814, 32.1033325, -40.9116898, 41.8718643

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1975069, upper bound: 60.2060877
time: 1.27 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1886302, upper bound: 60.1959966
time: 0.89 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1893149, upper bound: 60.1946898
time: 1.00 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1992069, upper bound: 60.2061578
time: 0.90 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7.6358347, 24.1759071, -8.4977674, 26.8933430, -34.5291786, 32.6736679
1: -10.7233191, 25.1112423, -11.8817339, 27.8938255, -38.6171455, 36.9929733
2: -9.2092772, 28.1248932, -10.1987410, 31.2466660, -40.4559441, 38.3236351
3: -10.1858873, 36.0929527, -11.3400717, 40.0894585, -50.2753410, 47.4330215
4: -8.8083553, 33.4915848, -9.7117977, 37.1281395, -45.9364929, 43.2033768

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1975069, upper bound: 60.2077988
time: 0.95 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1886302, upper bound: 60.1959966
time: 0.84 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1893149, upper bound: 60.1946898
time: 0.97 seconds

## Relational analysis of NS_B2_B2_A1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B2_B2_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1992069, upper bound: 60.2095409
time: 1.09 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -10.8226824, 33.9893799, -7.1544199, 23.2220268, -34.0447083, 41.1437950
1: -15.3359919, 35.2548904, -10.1142340, 24.1331444, -39.4691315, 45.3691216
2: -13.1706419, 39.2478790, -8.7066746, 27.0448685, -40.2155113, 47.9545517
3: -14.4467955, 50.4317055, -9.6192160, 34.7037201, -49.1505165, 60.0509224
4: -12.3785849, 46.6736145, -8.3802814, 32.1033325, -44.4819183, 55.0538940

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1988447, upper bound: 60.2078881
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1996536, upper bound: 60.2068885
time: 0.93 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2002825, upper bound: 60.2080188
time: 0.82 seconds

## BFS NS instance: NS_B2_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -10.8226824, 33.9893799, -8.4977674, 26.8933430, -37.7160225, 42.4871445
1: -15.3359919, 35.2548904, -11.8817339, 27.8938255, -43.2298164, 47.1366081
2: -13.1706419, 39.2478790, -10.1987410, 31.2466660, -44.4173050, 49.4466209
3: -14.4467955, 50.4317055, -11.3400717, 40.0894585, -54.5362549, 61.7717781
4: -12.3785849, 46.6736145, -9.7117977, 37.1281395, -49.5067253, 56.3854103

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1988447, upper bound: 60.2164317
time: 0.89 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1933523, upper bound: 60.2027758
time: 0.81 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## BFS NS instance: NS_B2_B2_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -11.6891575, 36.3419762, -7.1544199, 23.2220268, -34.9111862, 43.4963951
1: -16.4537983, 37.6537743, -10.1142340, 24.1331444, -40.5869408, 47.7680092
2: -14.1387310, 41.9565811, -8.7066746, 27.0448685, -41.1835976, 50.6632538
3: -15.5197172, 53.9093857, -9.6192160, 34.7037201, -50.2234383, 63.5286026
4: -13.2439423, 49.9412079, -8.3802814, 32.1033325, -45.3472748, 58.3214874

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1989648, upper bound: 60.2096593
time: 1.00 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1933784, upper bound: 60.2048802
time: 0.96 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1971996, upper bound: 60.2090099
time: 1.12 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_B2_B2_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -11.6891575, 36.3419762, -8.4977674, 26.8933430, -38.5825005, 44.8397446
1: -16.4537983, 37.6537743, -11.8817339, 27.8938255, -44.3476257, 49.5354996
2: -14.1387310, 41.9565811, -10.1987410, 31.2466660, -45.3853874, 52.1553230
3: -15.5197172, 53.9093857, -11.3400717, 40.0894585, -55.6091766, 65.2494583
4: -13.2439423, 49.9412079, -9.7117977, 37.1281395, -50.3720818, 59.6530075

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1989648, upper bound: 60.2189341
time: 1.21 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1933784, upper bound: 60.2053339
time: 1.01 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1971996, upper bound: 60.2218103
time: 1.10 seconds

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_B2_B2_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -9.4499655, 30.1438122, -10.9193020, 34.4716988, -43.9216652, 41.0631142
1: -13.4802599, 31.2940292, -15.4629669, 35.7357826, -49.2160378, 46.7569962
2: -11.5837431, 34.8744049, -13.2871695, 39.8115616, -51.3953018, 48.1615753
3: -12.6736584, 44.9068756, -14.5526495, 51.1244621, -63.7981186, 59.4595261
4: -10.9627438, 41.4599915, -12.4893827, 47.2725906, -58.2353363, 53.9493675

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## BFS NS instance: NS_B2_B2_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -9.4499655, 30.1438122, -14.8790970, 46.2554855, -55.7054520, 45.0229073
1: -13.4802599, 31.2940292, -21.0219193, 47.8620033, -61.3422546, 52.3159485
2: -11.5837431, 34.8744049, -18.0474167, 53.2342529, -64.8179932, 52.9218178
3: -12.6736584, 44.9068756, -19.7871208, 68.2334290, -80.9070892, 64.6939926
4: -10.9627438, 41.4599915, -16.7340488, 63.2242813, -74.1870193, 58.1940384

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

## BFS NS instance: NS_B2_B2_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -9.4896736, 30.2163372, -10.9193020, 34.4716988, -43.9613647, 41.1356354
1: -13.5354614, 31.3710442, -15.4629669, 35.7357826, -49.2712440, 46.8340111
2: -11.6318226, 34.9533844, -13.2871695, 39.8115616, -51.4433823, 48.2405548
3: -12.7202435, 45.0115929, -14.5526495, 51.1244621, -63.8447037, 59.5642433
4: -11.0013351, 41.5614967, -12.4893827, 47.2725906, -58.2739258, 54.0508728

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_B2_B2_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -9.4896736, 30.2163372, -14.8790970, 46.2554855, -55.7451515, 45.0954285
1: -13.5354614, 31.3710442, -21.0219193, 47.8620033, -61.3974648, 52.3929634
2: -11.6318226, 34.9533844, -18.0474167, 53.2342529, -64.8660736, 53.0007973
3: -12.7202435, 45.0115929, -19.7871208, 68.2334290, -80.9536743, 64.7987061
4: -11.0013351, 41.5614967, -16.7340488, 63.2242813, -74.2256165, 58.2955475

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_B2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.2577648, 38.2416573, -10.7720881, 34.0465012, -46.3042641, 49.0137444
1: -17.3925800, 39.6325226, -15.2580109, 35.2976646, -52.6902466, 54.8905334
2: -14.9581261, 44.1133270, -13.1122322, 39.3278351, -54.2859535, 57.2255592
3: -16.3262615, 56.6815491, -14.3589106, 50.5132675, -66.8395309, 71.0404587
4: -13.9751301, 52.4655342, -12.3350105, 46.6977806, -60.6729088, 64.8005447

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1947210, upper bound: 60.2195252
time: 0.82 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1983489, upper bound: 60.2221133
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.5875578, 42.2157211, -10.9193020, 34.4716988, -48.0592575, 53.1350250
1: -19.2908497, 43.7149506, -15.4629669, 35.7357826, -55.0266266, 59.1779175
2: -16.5635109, 48.6162186, -13.2871695, 39.8115616, -56.3750725, 61.9033890
3: -18.0938206, 62.4278107, -14.5526495, 51.1244621, -69.2182846, 76.9804611
4: -15.3811779, 57.8215942, -12.4893827, 47.2725906, -62.6537704, 70.3109741

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1999564, upper bound: 60.2195542
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2035843, upper bound: 60.2220668
time: 1.08 seconds

## BFS NS instance: NS_B2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.4853077, 41.9388199, -14.8790970, 46.2554855, -59.7407913, 56.8179169
1: -19.1443882, 43.4278259, -21.0219193, 47.8620033, -67.0063934, 64.4497452
2: -16.4361420, 48.3047523, -18.0474167, 53.2342529, -69.6703873, 66.3521729
3: -17.9578171, 62.0264702, -19.7871208, 68.2334290, -86.1912460, 81.8135834
4: -15.2681208, 57.4445114, -16.7340488, 63.2242813, -78.4923935, 74.1785583

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.5360928, 42.0253830, -14.8790970, 46.2554855, -59.7915802, 56.9044800
1: -19.2167587, 43.5229645, -21.0219193, 47.8620033, -67.0787506, 64.5448837
2: -16.5004272, 48.4057198, -18.0474167, 53.2342529, -69.7346802, 66.4531403
3: -18.0160503, 62.1467667, -19.7871208, 68.2334290, -86.2494812, 81.9338837
4: -15.3201580, 57.5685463, -16.7340488, 63.2242813, -78.5444412, 74.3025970

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.2605543, 26.1002769, -9.4017534, 29.9408741, -38.2014275, 35.5020294
1: -11.6701870, 27.1455231, -13.3599358, 31.0695763, -42.7397614, 40.5054588
2: -10.0508842, 30.4125671, -11.5086308, 34.6749001, -44.7257805, 41.9211884
3: -10.9955826, 38.9221077, -12.5667181, 44.5698586, -55.5654411, 51.4888191
4: -9.5514736, 36.0904274, -10.8749142, 41.1993904, -50.7508621, 46.9653397

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1927293, upper bound: 60.1927293
time: 0.93 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1927293, upper bound: 60.2004691
time: 0.86 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.7334709, 30.9533539, -7.9651947, 25.1922607, -34.9257278, 38.9185371
1: -13.8223515, 32.1135483, -11.2546768, 26.2076550, -40.0300064, 43.3682251
2: -11.9020767, 35.8392563, -9.6989326, 29.3655586, -41.2676353, 45.5381889
3: -13.0118752, 46.0697556, -10.5948057, 37.5727043, -50.5845795, 56.6645546
4: -11.2337151, 42.5850677, -9.2302942, 34.8430061, -46.0767212, 51.8153572

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2015409, upper bound: 60.2098301
time: 0.98 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1996039, upper bound: 60.2070050
time: 0.87 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.7334709, 30.9533539, -9.4017534, 29.9408741, -39.6743393, 40.3551064
1: -13.8223515, 32.1135483, -13.3599358, 31.0695763, -44.8919296, 45.4734840
2: -11.9020767, 35.8392563, -11.5086308, 34.6749001, -46.5769768, 47.3478813
3: -13.0118752, 46.0697556, -12.5667181, 44.5698586, -57.5817337, 58.6364670
4: -11.2337151, 42.5850677, -10.8749142, 41.1993904, -52.4331055, 53.4599762

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2015409, upper bound: 60.2149905
time: 1.03 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1996039, upper bound: 60.2160479
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.9127254, 28.5410595, -14.4449558, 43.7330017, -52.6457253, 42.9860039
1: -12.6067314, 29.6007442, -19.9845181, 45.2449799, -57.8517113, 49.5852585
2: -10.8521481, 33.0609627, -17.1869736, 50.4479446, -61.3000946, 50.2479362
3: -11.9536343, 42.4839859, -19.0135098, 64.7775345, -76.7311630, 61.4974937
4: -10.2816601, 39.2907829, -16.0041656, 60.0903854, -70.3720322, 55.2949486

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2246249, upper bound: 60.2046087
time: 1.06 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250435, upper bound: 60.2050320
time: 1.02 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.0973234, 29.0640278, -15.6973810, 47.4907303, -56.5880547, 44.7614059
1: -12.8702602, 30.1397648, -21.8695736, 49.1121216, -61.9823799, 52.0093384
2: -11.0743856, 33.6567726, -18.7495499, 54.6521835, -65.7265701, 52.4063225
3: -12.1959352, 43.2404213, -20.6822147, 70.1567383, -82.3526764, 63.9226379
4: -10.4772635, 40.0022278, -17.3585587, 65.0967026, -75.5739441, 57.3607826

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075682, upper bound: 60.1974886
time: 1.02 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153710, upper bound: 60.2014285
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2239678, upper bound: 60.2132470
time: 1.08 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.6968594, 33.9088364, -14.3939734, 43.6217346, -54.3185883, 48.3028030
1: -15.0916224, 35.1240158, -19.9141350, 45.1287689, -60.2203865, 55.0381508
2: -12.9737864, 39.2294922, -17.1281528, 50.3204918, -63.2942657, 56.3576431
3: -14.2583523, 50.4437637, -18.9586773, 64.6176758, -78.8760300, 69.4024353
4: -12.2118120, 46.6417999, -15.9529552, 59.9371452, -72.1489563, 62.5947571

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2244763, upper bound: 60.2056502
time: 1.02 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274379, upper bound: 60.2079513
time: 1.04 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.8696327, 34.4083939, -15.6444416, 47.3736610, -58.2432938, 50.0528336
1: -15.3343544, 35.6384964, -21.7951660, 48.9891968, -64.3235474, 57.4336586
2: -13.1785946, 39.7972412, -18.6808281, 54.5172272, -67.6958237, 58.4780655
3: -14.4879837, 51.1623955, -20.6248932, 69.9893723, -84.4773560, 71.7872925
4: -12.3919563, 47.3149300, -17.3050709, 64.9321213, -77.3240738, 64.6200027

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1902201, upper bound: 60.1812491
time: 1.11 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2253622, upper bound: 60.2174142
time: 0.97 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -14.4011230, 43.6175117, -6.5293684, 21.2264290, -35.6275520, 50.1468697
1: -19.9237652, 45.1279793, -9.1998091, 22.0716133, -41.9953766, 54.3277855
2: -17.1381760, 50.3091011, -7.9254265, 24.7797070, -41.9178848, 58.2345276
3: -18.9676342, 64.6169815, -8.7611837, 31.7485313, -50.7161636, 73.3781586
4: -15.9549217, 59.9325905, -7.6640325, 29.4305820, -45.3855057, 67.5966263

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A1

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1984449, upper bound: 60.2169194
time: 1.16 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A1_A2

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2008925, upper bound: 60.2197015
time: 0.80 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -15.6426764, 47.3275642, -6.7173672, 21.7505741, -37.3932381, 54.0449295
1: -21.7922020, 48.9486465, -9.4661121, 22.6145306, -44.4067307, 58.4147568
2: -18.6856174, 54.4668770, -8.1495190, 25.3800755, -44.0656853, 62.6163902
3: -20.6187820, 69.9195786, -9.0071821, 32.5074387, -53.1262169, 78.9267578
4: -17.2955723, 64.8848114, -7.8611236, 30.1524315, -47.4480057, 72.7459335

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1968857, upper bound: 60.2074572
time: 1.28 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073766, upper bound: 60.2168155
time: 0.98 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -13.3044767, 40.1446075, -11.9908962, 36.5895920, -49.8940697, 52.1355057
1: -18.4690800, 41.5440445, -16.6667862, 37.8833618, -56.3524399, 58.2108307
2: -15.8064594, 46.3984032, -14.2192478, 42.3016624, -58.1081161, 60.6176529
3: -17.5225220, 59.2194366, -15.8255281, 54.0901794, -71.6126785, 75.0449677
4: -14.6301203, 55.2837143, -13.2623749, 50.4039803, -65.0341034, 68.5460892

Time for backsubstitution: 0.83 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.98 + 417.21 = 420.19 seconds
