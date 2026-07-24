## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.52160481426


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-25.7190342, 27.5826702, -25.7190342, 27.5826702, -53.3016968, 53.3016968)
1: (-199.8423004, 64.8683853, -199.8423004, 64.8683853, -264.7106934, 264.7106934)
2: (-107.3599548, 59.5958939, -107.3599548, 59.5958939, -166.9558411, 166.9558411)
3: (-139.0563507, 47.8273277, -139.0563507, 47.8273277, -186.8836365, 186.8836365)
4: (-75.8289719, 50.8402481, -75.8289719, 50.8402481, -126.6692200, 126.6692200)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.94 + 2.18 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -42.5258574, upper bound: 42.5258574

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250534, upper bound: 42.5251229
time: 0.82 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250534, upper bound: 42.5254135
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -42.5250534, upper bound: 42.5251229
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -42.5250534, upper bound: 42.5254135

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -24.8010502, 26.5595016, -25.0954666, 26.8886642, -51.6897049, 51.6549683
1: -192.5416412, 62.5583725, -194.8900604, 63.2984962, -255.8401337, 257.4484253
2: -103.6234741, 57.3975601, -104.8174820, 58.1028900, -161.7263489, 162.2150421
3: -134.0417786, 46.0722160, -135.6530304, 46.6357918, -180.6775665, 181.7252502
4: -73.1696014, 48.9217453, -74.0216293, 49.5378914, -122.7074738, 122.9433670

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249411, upper bound: 42.5248646
time: 0.80 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245013, upper bound: 42.5247364
time: 0.74 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -26.7943859, 28.9738884, -24.9679661, 26.9112587, -53.7056427, 53.9418564
1: -210.1424103, 67.7370834, -195.1482239, 63.0032120, -273.1455994, 262.8853149
2: -111.8599854, 62.6114197, -104.4212341, 58.0235748, -169.8835602, 167.0326538
3: -145.8540802, 50.1733856, -135.6281128, 46.5875702, -192.4416504, 185.8014984
4: -79.0120392, 53.4659767, -73.6780853, 49.5443764, -128.5564117, 127.1440582

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253922, upper bound: 42.5252362
time: 0.93 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253926, upper bound: 42.5253926
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.79 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -42.5249411, upper bound: 42.5248646
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -42.5245013, upper bound: 42.5247364
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -42.5253922, upper bound: 42.5252362
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -42.5253926, upper bound: 42.5253926

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -24.4799995, 26.2047539, -24.9978771, 26.7808342, -51.2608299, 51.2026291
1: -189.9776306, 61.7499619, -194.1100006, 63.0530243, -253.0306244, 255.8599548
2: -102.3120575, 56.6364441, -104.4160995, 57.8717957, -160.1838531, 161.0524902
3: -132.2732391, 45.4600182, -135.1147308, 46.4498749, -178.7231140, 180.5747528
4: -72.2445068, 48.2562027, -73.7400513, 49.3357162, -121.5802155, 121.9962540

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248207, upper bound: 42.5247444
time: 0.84 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247815, upper bound: 42.5246807
time: 0.73 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -24.6087227, 26.4229355, -56.0007896, 56.3389091
1: -229.0471802, 74.9232941, -191.6903534, 62.1005630, -291.1477356, 266.6136169
2: -123.3018036, 68.8398743, -102.8633575, 57.0567818, -180.3585815, 171.7032318
3: -159.7204285, 55.1654396, -133.3197327, 45.8182373, -205.5386658, 188.4851685
4: -87.1864471, 58.7552109, -72.6237869, 48.6401939, -135.8266449, 131.3789978

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
time: 0.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -26.2792683, 28.3905525, -23.9213810, 25.7282238, -52.0074921, 52.3119316
1: -205.9583282, 66.4369202, -186.7190704, 60.3652573, -266.3235779, 253.1559906
2: -109.7402191, 61.3630714, -100.1371613, 55.4838181, -165.2240295, 161.5002289
3: -142.9961395, 49.1773376, -129.8739014, 44.5728569, -187.5690002, 179.0512390
4: -77.5196533, 52.3775101, -70.6575394, 47.3313484, -124.8509979, 123.0350418

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253393, upper bound: 42.5251939
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251735, upper bound: 42.5251129
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5252968, upper bound: 42.5252067
time: 0.79 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -25.5069523, 27.6650848, -28.0468731, 30.6064281, -56.1133804, 55.7119598
1: -199.0617065, 64.5533600, -216.8084106, 70.9022369, -269.9639282, 281.3617554
2: -106.1441345, 59.6122742, -115.5001678, 65.8244019, -171.9685364, 175.1124268
3: -138.3128967, 47.8330345, -150.5123138, 52.7580070, -191.0708771, 198.3453522
4: -75.1521225, 51.0435257, -82.0402298, 56.7826653, -131.9347687, 133.0837402

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253153, upper bound: 42.5253396
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253313, upper bound: 42.5253313
time: 1.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.99 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5248207, upper bound: 42.5247444
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5247815, upper bound: 42.5246807
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5251735, upper bound: 42.5251129
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5252968, upper bound: 42.5252067
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5253153, upper bound: 42.5253396
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -42.5253313, upper bound: 42.5253313

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -22.6538467, 24.3994389, -24.3170261, 26.0196419, -48.6734848, 48.7164650
1: -177.5626831, 57.2307472, -188.7777100, 61.3393555, -238.9020233, 246.0084534
2: -94.8510818, 52.6801071, -101.6430054, 56.2420082, -151.0930634, 154.3231201
3: -123.4291611, 42.3136826, -131.4544220, 45.1400719, -168.5692291, 173.7680969
4: -66.8511047, 44.8885117, -71.7761002, 47.9009399, -114.7520447, 116.6646042

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247648, upper bound: 42.5247211
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248207, upper bound: 42.5247444
time: 0.83 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -23.4168110, 25.1101437, -24.2556438, 26.0264816, -49.4432831, 49.3657875
1: -182.2654877, 58.9939041, -188.8668060, 61.1194038, -243.3848877, 247.8607178
2: -97.9469604, 54.1482048, -101.4247589, 56.1455803, -154.0925293, 155.5729523
3: -126.7982788, 43.5060692, -131.3758240, 45.0925522, -171.8908386, 174.8818970
4: -69.0682983, 46.1689339, -71.5257950, 47.8870583, -116.9553528, 117.6947174

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247183, upper bound: 42.5245417
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242058, upper bound: 42.5243339
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -24.2893944, 26.0702209, -55.6480751, 56.0195808
1: -229.0471802, 74.9232941, -189.1606445, 61.2974892, -290.3446350, 264.0839233
2: -123.3018036, 68.8398743, -101.5628204, 56.2950630, -179.5968628, 170.4026794
3: -159.7204285, 55.1654396, -131.5801086, 45.2099495, -204.9303589, 186.7455444
4: -87.1864471, 58.7552109, -71.6991501, 47.9794426, -135.1658936, 130.4543610

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
time: 0.77 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
time: 0.73 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -26.3102512, 28.5070992, -58.0849533, 58.0404358
1: -229.0471802, 74.9232941, -207.0109100, 66.5537033, -295.6008911, 281.9341431
2: -123.3018036, 68.8398743, -109.9704285, 61.5709267, -184.8727264, 178.8103027
3: -159.7204285, 55.1654396, -143.5694733, 49.3592339, -209.0796204, 198.7348785
4: -87.1864471, 58.7552109, -77.6336136, 52.5638924, -139.7503357, 136.3888245

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5246817
time: 0.78 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -24.4880104, 26.6191750, -23.2508087, 24.9838295, -49.4718399, 49.8699799
1: -193.9053040, 62.0593262, -181.5227203, 58.6806030, -252.5858765, 243.5820465
2: -102.5959549, 57.4524193, -97.4192200, 53.8867798, -156.4826965, 154.8716431
3: -134.4172211, 46.0871239, -126.2996674, 43.2886391, -177.7058563, 172.3867950
4: -72.3312836, 48.9944077, -68.7272186, 45.9235611, -118.2548370, 117.7216263

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251668, upper bound: 42.5251129
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251668, upper bound: 42.5251129
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.3592663, 27.3998585, -23.3312092, 25.1120033, -50.4712639, 50.7310677
1: -198.8013000, 64.0780869, -182.3369446, 58.8451843, -257.6464539, 246.4150391
2: -105.8910065, 59.1699066, -97.7210693, 54.1002197, -159.9912262, 156.8909760
3: -137.9783783, 47.4518242, -126.7664948, 43.4863739, -181.4647369, 174.2183228
4: -74.7712250, 50.5302086, -68.8826828, 46.1696205, -120.9408264, 119.4128723

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5252191, upper bound: 42.5251675
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5252191, upper bound: 42.5252067
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -25.1285477, 27.2507038, -27.9397202, 30.4875107, -55.6160507, 55.1904106
1: -196.0667725, 63.6016159, -215.9494629, 70.6298599, -266.6966248, 279.5510864
2: -104.5841293, 58.7149239, -115.0580292, 65.5666885, -170.1508179, 173.7729492
3: -136.2496948, 47.1125641, -149.9244690, 52.5517159, -188.8014069, 197.0370178
4: -74.0659866, 50.2596321, -81.7292862, 56.5573845, -130.6233673, 131.9889221

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251553, upper bound: 42.5251740
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248204, upper bound: 42.5249802
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -27.7842331, 30.2820053, -27.4250126, 29.9815845, -57.7658043, 57.7070160
1: -217.0359039, 70.3626480, -212.3530579, 69.3500748, -286.3859863, 282.7156677
2: -115.2777176, 65.2840881, -112.9645309, 64.4366608, -179.7143860, 178.2486267
3: -150.5507812, 52.3172340, -147.3403015, 51.6590881, -202.2098694, 199.6575317
4: -81.6127853, 56.1550827, -80.2352066, 55.5939445, -137.2067261, 136.3902740

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250047, upper bound: 42.5248620
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246724, upper bound: 42.5247885
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.53 seconds
NS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5247648, upper bound: 42.5247211
NS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5248207, upper bound: 42.5247444
NS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5247183, upper bound: 42.5245417
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5242058, upper bound: 42.5243339
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5246817
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5244219, upper bound: 42.5244219
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5251668, upper bound: 42.5251129
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5251668, upper bound: 42.5251129
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5252191, upper bound: 42.5251675
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5252191, upper bound: 42.5252067
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5251553, upper bound: 42.5251740
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5248204, upper bound: 42.5249802
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5250047, upper bound: 42.5248620
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -42.5246724, upper bound: 42.5247885

## BFS NS instance: NS_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -23.6676579, 25.2899132, -46.0723343, 45.9559097
1: -162.5009918, 52.2805252, -183.5494843, 59.6936836, -222.1946716, 235.8299866
2: -86.8753586, 48.2073059, -98.9608612, 54.6737518, -141.5491028, 147.1681366
3: -112.7218323, 38.6586380, -127.8776093, 43.8820724, -156.6038818, 166.5362091
4: -61.1049271, 41.0587082, -69.8856354, 46.5340462, -107.6389771, 110.9443436

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -24.1708641, 25.8645039, -48.1744881, 48.2024918
1: -175.0307159, 56.3754234, -187.7192078, 60.9674072, -235.9981232, 244.0945892
2: -93.4728622, 51.8776817, -101.0567093, 55.8955574, -149.3684082, 152.9343567
3: -121.6609573, 41.6777878, -130.7156525, 44.8674316, -166.5283813, 172.3934174
4: -65.8667603, 44.1967926, -71.3568039, 47.6022873, -113.4690247, 115.5535889

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
time: 0.80 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
time: 0.75 seconds

## BFS NS instance: NS_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -24.4373074, 26.1575623, -48.8155212, 48.6913681
1: -176.2945251, 57.0708656, -191.2181091, 61.7332077, -238.0277405, 248.2889557
2: -94.9132462, 52.2951469, -102.8004684, 56.6318398, -151.5450745, 155.0955658
3: -122.6806107, 42.0384216, -133.0458069, 45.4343948, -168.1149902, 175.0842285
4: -66.9144669, 44.5387917, -72.3423996, 48.1096687, -115.0241394, 116.8811951

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.70 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.84 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -23.2207546, 24.8942413, -23.8525772, 25.5809040, -48.8016586, 48.7468185
1: -180.7801514, 58.5105705, -185.8290710, 60.1264229, -240.9065552, 244.3396149
2: -97.1722260, 53.6857529, -99.8393784, 55.1936913, -152.3659058, 153.5251312
3: -125.7768021, 43.1376152, -129.2871399, 44.3338737, -170.1106720, 172.4247589
4: -68.5218201, 45.7584648, -70.4036942, 47.0381012, -115.5599213, 116.1621552

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5242761
time: 0.82 seconds

## Relational analysis of NS_A1_A1_A2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5243339
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -24.4799995, 26.2047539, -55.7826080, 56.2101860
1: -229.0471802, 74.9232941, -189.9776306, 61.7499619, -290.7971497, 264.9008789
2: -123.3018036, 68.8398743, -102.3120575, 56.6364441, -179.9382477, 171.1519318
3: -159.7204285, 55.1654396, -132.2732391, 45.4600182, -205.1804504, 187.4386749
4: -87.1864471, 58.7552109, -72.2445068, 48.2562027, -135.4426575, 130.9996948

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -29.5778542, 31.7301865, -61.3080406, 61.3080406
1: -229.0471802, 74.9232941, -229.0471802, 74.9232941, -303.9704285, 303.9704590
2: -123.3018036, 68.8398743, -123.3018036, 68.8398743, -192.1416779, 192.1416779
3: -159.7204285, 55.1654396, -159.7204285, 55.1654396, -214.8858490, 214.8858337
4: -87.1864471, 58.7552109, -87.1864471, 58.7552109, -145.9416504, 145.9416504

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -26.4212627, 28.5676899, -58.1455460, 58.1514473
1: -229.0471802, 74.9232941, -207.2078552, 66.7981491, -295.8453064, 282.1311340
2: -123.3018036, 68.8398743, -110.3238068, 61.7285576, -185.0303650, 179.1636810
3: -159.7204285, 55.1654396, -143.8279572, 49.4672852, -209.1877136, 198.9933777
4: -87.1864471, 58.7552109, -77.9395523, 52.6980896, -139.8845367, 136.6947632

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5232182, upper bound: 42.5242308
time: 0.80 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -29.5778542, 31.7301865, -31.5138721, 34.0930443, -63.6708946, 63.2440567
1: -229.0471802, 74.9232941, -246.2247620, 79.9593048, -309.0064697, 321.1480408
2: -123.3018036, 68.8398743, -131.3201141, 73.9421082, -197.2439117, 200.1599884
3: -159.7204285, 55.1654396, -171.2086182, 59.1738014, -218.8942108, 226.3740234
4: -87.1864471, 58.7552109, -92.8589935, 63.2144089, -150.4008484, 151.6141968

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5232182, upper bound: 42.5242308
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -23.9437981, 26.0091267, -23.2508087, 24.9838295, -48.9276276, 49.2599297
1: -189.5855560, 60.6800919, -181.5227203, 58.6806030, -248.2661133, 242.2028198
2: -100.3765869, 56.1372452, -97.4192200, 53.8867798, -154.2633514, 153.5564575
3: -131.4502869, 45.0423508, -126.2996674, 43.2886391, -174.7389221, 171.3420105
4: -70.7577896, 47.8496780, -68.7272186, 45.9235611, -116.6813354, 116.5768967

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251088
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251088
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -28.9298801, 31.6730061, -23.2508087, 24.9838295, -53.9137115, 54.9238091
1: -224.3556976, 72.9705429, -181.5227203, 58.6806030, -283.0363159, 254.4932556
2: -118.4201279, 68.1376343, -97.4192200, 53.8867798, -172.3068695, 165.5568542
3: -155.3976288, 54.6253014, -126.2996674, 43.2886391, -198.6862640, 180.9249725
4: -84.1458130, 58.8627205, -68.7272186, 45.9235611, -130.0693665, 127.5899353

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251088
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251129
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.3592663, 27.3998585, -22.0516434, 23.8416100, -49.2008743, 49.4514999
1: -198.8013000, 64.0780869, -173.4043884, 55.7399139, -254.5411682, 237.4824829
2: -105.8910065, 59.1699066, -92.3806000, 51.3627319, -157.2537384, 151.5505066
3: -137.9783783, 47.4518242, -120.4406128, 41.2896156, -179.2679749, 167.8924103
4: -74.7712250, 50.5302086, -65.0942993, 43.8127327, -118.5839462, 115.6244965

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251827, upper bound: 42.5247197
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5164087, upper bound: 42.5156022
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250426, upper bound: 42.5249876
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.3592663, 27.3998585, -22.9923649, 24.7495518, -50.1088181, 50.3922234
1: -198.8013000, 64.0780869, -179.6977234, 57.9785271, -256.7798157, 243.7758179
2: -105.8910065, 59.1699066, -96.2843094, 53.2970352, -159.1880493, 155.4542236
3: -137.9783783, 47.4518242, -124.9043350, 42.8572655, -180.8356476, 172.3561554
4: -74.7712250, 50.5302086, -67.8521957, 45.4986076, -120.2698135, 118.3823853

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251827, upper bound: 42.5247206
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251420, upper bound: 42.5252067
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251420, upper bound: 42.5252067
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -24.4496040, 26.4896317, -28.8858795, 31.4754639, -55.9250679, 55.3755112
1: -190.7049103, 61.8814011, -225.3232727, 73.3083801, -264.0133057, 287.2046204
2: -101.8645096, 57.0572319, -120.0000381, 67.9255066, -169.7899933, 177.0572357
3: -132.5478821, 45.7986298, -156.5441437, 54.4177704, -186.9656525, 202.3427582
4: -72.1410828, 48.8005486, -85.0120926, 58.2711563, -130.4122314, 133.8126373

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5006193, upper bound: 42.5003087
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.4890875, upper bound: 42.4903275
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -24.9405613, 27.0451431, -27.5098724, 30.0182037, -54.9587631, 54.5550117
1: -194.6311493, 63.1357002, -212.6029053, 69.5579453, -264.1890869, 275.7385864
2: -103.8309555, 58.2747116, -113.3292847, 64.5503006, -168.3812561, 171.6040039
3: -135.2598267, 46.7586823, -147.6370392, 51.7370567, -186.9968872, 194.3957062
4: -73.5375595, 49.8711929, -80.5066833, 55.6678314, -129.2053833, 130.3778229

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5141008, upper bound: 42.5139239
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5083991, upper bound: 42.5090196
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -26.8686619, 29.2946320, -28.2844448, 30.8601799, -57.7288437, 57.5790787
1: -209.9876099, 68.0620193, -220.9116669, 71.8195343, -281.8071289, 288.9736328
2: -111.5995865, 63.1134796, -117.5514755, 66.5724182, -178.1719818, 180.6649323
3: -145.6613007, 50.5970116, -153.4175873, 53.3464661, -199.0077667, 204.0146027
4: -79.0149994, 54.2718697, -83.2847214, 57.1096497, -136.1246490, 137.5565796

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5054409, upper bound: 42.5088556
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248577, upper bound: 42.5246838
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -27.6047096, 30.0823708, -26.9718399, 29.4914570, -57.0961647, 57.0542107
1: -215.6543579, 69.9152908, -208.8011932, 68.2165451, -283.8708496, 278.7164612
2: -114.5671692, 64.8554840, -111.1173325, 63.3720474, -177.9391785, 175.9727936
3: -149.6035156, 51.9750519, -144.8974915, 50.8023949, -200.4059143, 196.8725433
4: -81.1077194, 55.7740860, -78.9380112, 54.6672401, -135.7749634, 134.7120667

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5133999, upper bound: 42.5117209
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5066982, upper bound: 42.5066982
time: 1.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.31 seconds
NS_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
NS_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
NS_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
NS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
NS_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5242761
NS_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5243339
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251088
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251088
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251088
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5251342, upper bound: 42.5251129
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5164087, upper bound: 42.5156022
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5250426, upper bound: 42.5249876
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5251420, upper bound: 42.5252067
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5251420, upper bound: 42.5252067
NS_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5006193, upper bound: 42.5003087
NS_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.4890875, upper bound: 42.4903275
NS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5141008, upper bound: 42.5139239
NS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5083991, upper bound: 42.5090196
NS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5054409, upper bound: 42.5088556
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5248577, upper bound: 42.5246838
NS_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5133999, upper bound: 42.5117209
NS_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 0, lower bound: -42.5066982, upper bound: 42.5066982

## BFS NS instance: NS_A1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -23.3777924, 24.9646358, -45.7470512, 45.6660461
1: -162.5009918, 52.2805252, -181.1986084, 58.9628792, -221.4638519, 233.4791260
2: -86.8753586, 48.2073059, -97.7774506, 53.9757538, -140.8511047, 145.9847107
3: -112.7218323, 38.6586380, -126.2686005, 43.3240814, -156.0459137, 164.9272461
4: -61.1049271, 41.0587082, -69.0422897, 45.9265709, -107.0314941, 110.1009979

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.86 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -25.5274906, 27.5504303, -48.3328476, 47.8157425
1: -162.5009918, 52.2805252, -199.9503326, 64.5555801, -227.0565796, 252.2308350
2: -86.8753586, 48.2073059, -106.6491318, 59.5688515, -146.4442139, 154.8564453
3: -112.7218323, 38.6586380, -138.8906097, 47.7235718, -160.4454041, 177.5492249
4: -61.1049271, 41.0587082, -75.3524628, 50.7957573, -111.9006805, 116.4111710

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.82 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5247211
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5247211
time: 1.00 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -23.8783913, 25.5370865, -47.8470726, 47.9100189
1: -175.0307159, 56.3754234, -185.3728333, 60.2317009, -235.2623901, 241.7482147
2: -93.4728622, 51.8776817, -99.8659286, 55.1942024, -148.6670685, 151.7435913
3: -121.6609573, 41.6777878, -129.1079865, 44.3064613, -165.9674225, 170.7857666
4: -65.8667603, 44.1967926, -70.5088425, 46.9901390, -112.8568878, 114.7056198

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
time: 0.88 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -25.9748001, 28.0639954, -50.3739738, 50.0064278
1: -175.0307159, 56.3754234, -203.7057495, 65.6859207, -240.7166290, 260.0811462
2: -93.4728622, 51.8776817, -108.5201035, 60.6610031, -154.1338654, 160.3977814
3: -121.6609573, 41.6777878, -141.4412384, 48.6030922, -170.2640533, 183.1190186
4: -65.8667603, 44.1967926, -76.6605072, 51.7508049, -117.6175690, 120.8572769

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5242959
time: 0.81 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
time: 0.83 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -24.1344643, 25.8231010, -48.4810638, 48.3885231
1: -176.2945251, 57.0708656, -188.8656006, 60.9710121, -237.2655334, 245.9364624
2: -94.9132462, 52.2951469, -101.5838699, 55.9076805, -150.8209229, 153.8790131
3: -122.6806107, 42.0384216, -131.4313049, 44.8561783, -167.5367737, 173.4697266
4: -66.9144669, 44.5387917, -71.4716873, 47.4777031, -114.3921509, 116.0104675

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -26.4863796, 28.6215286, -51.2794876, 50.7404404
1: -176.2945251, 57.0708656, -208.7827454, 67.0635452, -243.3580627, 265.8535767
2: -94.9132462, 52.2951469, -111.1408081, 61.9564056, -156.8696442, 163.4359436
3: -122.6806107, 42.0384216, -144.8793488, 49.6145172, -172.2951050, 186.9177704
4: -66.9144669, 44.5387917, -78.3303452, 52.7650566, -119.6795197, 122.8691406

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5245417
time: 0.75 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.89 seconds

## BFS NS instance: NS_A1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -23.5677090, 25.2059841, -23.8525772, 25.5809040, -49.1486130, 49.0585632
1: -184.3307343, 59.5229073, -185.8290710, 60.1264229, -244.4571381, 245.3519745
2: -99.1971664, 54.5543060, -99.8393784, 55.1936913, -154.3908234, 154.3936768
3: -128.2790833, 43.7834854, -129.2871399, 44.3338737, -172.6129456, 173.0706177
4: -69.7876740, 46.3303642, -70.4036942, 47.0381012, -116.8257751, 116.7340469

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240240, upper bound: 42.5242638
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5242761
time: 0.87 seconds

## Relational analysis of NS_A1_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5242761
time: 0.86 seconds

## BFS NS instance: NS_A1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -23.0098019, 24.6616135, -23.8525772, 25.5809040, -48.5907021, 48.5141907
1: -179.2008820, 57.9912300, -185.8290710, 60.1264229, -239.3273010, 243.8202667
2: -96.3431702, 53.1889877, -99.8393784, 55.1936913, -151.5368347, 153.0283508
3: -124.6888733, 42.7408600, -129.2871399, 44.3338737, -169.0227203, 172.0279846
4: -67.9345169, 45.3148346, -70.4036942, 47.0381012, -114.9726181, 115.7185287

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5243339
time: 0.83 seconds

## Relational analysis of NS_A1_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5243339
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -23.9437981, 26.0091267, -22.0815239, 23.8715057, -47.8153038, 48.0906448
1: -189.5855560, 60.6800919, -173.6061249, 55.8173485, -245.4028625, 234.2861938
2: -100.3765869, 56.1372452, -92.5020294, 51.4314804, -151.8080750, 148.6392822
3: -131.4502869, 45.0423508, -120.5883484, 41.3429527, -172.7932434, 165.6307068
4: -70.7577896, 47.8496780, -65.1844482, 43.8699837, -114.6277695, 113.0341263

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5155249, upper bound: 42.5149689
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249514, upper bound: 42.5249514
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -23.9437981, 26.0091267, -22.9923649, 24.7495518, -48.6933517, 49.0014915
1: -189.5855560, 60.6800919, -179.6977234, 57.9785271, -247.5640717, 240.3778076
2: -100.3765869, 56.1372452, -96.2843094, 53.2970352, -153.6736145, 152.4215546
3: -131.4502869, 45.0423508, -124.9043350, 42.8572655, -174.3075562, 169.9466858
4: -70.7577896, 47.8496780, -67.8521957, 45.4986076, -116.2563858, 115.7018661

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5149901, upper bound: 42.5155315
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249514, upper bound: 42.5249686
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -28.9298801, 31.6730061, -22.0815239, 23.8715057, -52.8013840, 53.7545319
1: -224.3556976, 72.9705429, -173.6061249, 55.8173485, -280.1730347, 246.5766449
2: -118.4201279, 68.1376343, -92.5020294, 51.4314804, -169.8516083, 160.6396637
3: -155.3976288, 54.6253014, -120.5883484, 41.3429527, -196.7405853, 175.2136536
4: -84.1458130, 58.8627205, -65.1844482, 43.8699837, -128.0157928, 124.0471649

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246851, upper bound: 42.5247195
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248778, upper bound: 42.5247712
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -28.9298801, 31.6730061, -22.9923649, 24.7495518, -53.6794319, 54.6653709
1: -224.3556976, 72.9705429, -179.6977234, 57.9785271, -282.3342285, 252.6682739
2: -118.4201279, 68.1376343, -96.2843094, 53.2970352, -171.7171631, 164.4219360
3: -155.3976288, 54.6253014, -124.9043350, 42.8572655, -198.2548981, 179.5296326
4: -84.1458130, 58.8627205, -67.8521957, 45.4986076, -129.6444244, 126.7149048

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246851, upper bound: 42.5247266
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248778, upper bound: 42.5247816
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.0404797, 27.0764561, -22.0516434, 23.8416100, -48.8820877, 49.1280975
1: -196.5489349, 63.2931900, -173.4043884, 55.7399139, -252.2888031, 236.6975708
2: -104.5705185, 58.4835167, -92.3806000, 51.3627319, -155.9332275, 150.8641205
3: -136.3712158, 46.8934326, -120.4406128, 41.2896156, -177.6608276, 167.3340149
4: -73.8378906, 49.9465103, -65.0942993, 43.8127327, -117.6506119, 115.0408096

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250381, upper bound: 42.5248628
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249772, upper bound: 42.5248630
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -24.8936996, 26.8752785, -22.9923649, 24.7495518, -49.6432495, 49.8676453
1: -194.9888306, 62.9003754, -179.6977234, 57.9785271, -252.9673615, 242.5980835
2: -103.9552460, 58.0430489, -96.2843094, 53.2970352, -157.2522583, 154.3273621
3: -135.3714142, 46.5535889, -124.9043350, 42.8572655, -178.2286835, 171.4579163
4: -73.4126205, 49.5542564, -67.8521957, 45.4986076, -118.9112244, 117.4064484

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5163349, upper bound: 42.5156165
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250306, upper bound: 42.5250309
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -29.3340988, 32.0556335, -22.9923649, 24.7495518, -54.0836487, 55.0479965
1: -226.6908875, 74.1457672, -179.6977234, 57.9785271, -284.6694031, 253.8434906
2: -120.3348007, 69.0446930, -96.2843094, 53.2970352, -173.6318207, 165.3290100
3: -157.2002106, 55.2548256, -124.9043350, 42.8572655, -200.0574799, 180.1591644
4: -85.5657196, 59.6204300, -67.8521957, 45.4986076, -131.0643311, 127.4726257

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240314, upper bound: 42.5237085
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238475, upper bound: 42.5236094
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -26.8686619, 29.2946320, -27.9369621, 30.5047836, -57.3734436, 57.2315941
1: -209.9876099, 68.0620193, -218.4352570, 70.9552307, -280.9428406, 286.4972229
2: -111.5995865, 63.1134796, -116.1204529, 65.8103790, -177.4099274, 179.2339172
3: -145.6613007, 50.5970116, -151.6511536, 52.7323990, -198.3937073, 202.2481689
4: -79.0149994, 54.2718697, -82.2626343, 56.4622269, -135.4772186, 136.5344696

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.4991350, upper bound: 42.4985913
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.4851749, upper bound: 42.4862170
time: 0.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.20 seconds
NS_A1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
NS_A1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
NS_A1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5247211
NS_A1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5247211
NS_A1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
NS_A1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
NS_A1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
NS_A1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5243162
NS_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5245417
NS_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5242761
NS_A1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5242761
NS_A1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5243339
NS_A1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5240409, upper bound: 42.5243339
NS_A2_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5155249, upper bound: 42.5149689
NS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5249514, upper bound: 42.5249514
NS_A2_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5149901, upper bound: 42.5155315
NS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5249514, upper bound: 42.5249686
NS_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5246851, upper bound: 42.5247195
NS_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5248778, upper bound: 42.5247712
NS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5246851, upper bound: 42.5247266
NS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5248778, upper bound: 42.5247816
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5250381, upper bound: 42.5248628
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5249772, upper bound: 42.5248630
NS_A2_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5163349, upper bound: 42.5156165
NS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5250306, upper bound: 42.5250309
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5240314, upper bound: 42.5237085
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.5238475, upper bound: 42.5236094
NS_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.4991350, upper bound: 42.4985913
NS_A2_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.20
Output dim: 0, lower bound: -42.4851749, upper bound: 42.4862170

## BFS NS instance: NS_A1_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -23.1517696, 24.7154026, -45.4978142, 45.4400215
1: -162.5009918, 52.2805252, -179.3949738, 58.3935814, -220.8945770, 231.6754608
2: -86.8753586, 48.2073059, -96.8556061, 53.4386330, -140.3139954, 145.0628967
3: -112.7218323, 38.6586380, -125.0260620, 42.8924789, -155.6143188, 163.6846771
4: -61.1049271, 41.0587082, -68.3917923, 45.4579697, -106.5628967, 109.4505005

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239105, upper bound: 42.5232180
time: 0.79 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -28.3382797, 30.3441219, -51.1265411, 50.6265335
1: -162.5009918, 52.2805252, -219.0886078, 71.7793350, -234.2803345, 271.3691406
2: -86.8753586, 48.2073059, -118.1774826, 65.8468475, -152.7221985, 166.3847656
3: -112.7218323, 38.6586380, -152.9013519, 52.7661552, -165.4879761, 191.5599670
4: -61.1049271, 41.0587082, -83.5728531, 56.1421547, -117.2470779, 124.6315613

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244737, upper bound: 42.5243222
time: 0.89 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239105, upper bound: 42.5232180
time: 0.75 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -25.3030243, 27.3072605, -48.0896721, 47.5912781
1: -162.5009918, 52.2805252, -198.1923065, 63.9914169, -226.4924011, 250.4728088
2: -86.8753586, 48.2073059, -105.7287598, 59.0384521, -145.9138184, 153.9360199
3: -112.7218323, 38.6586380, -137.6752319, 47.3000031, -160.0217896, 176.3338470
4: -61.1049271, 41.0587082, -74.7064972, 50.3358917, -111.4408188, 115.7652054

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240352, upper bound: 42.5234643
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -20.7824211, 22.2882519, -28.2243385, 30.6345634, -51.4169846, 50.5125885
1: -162.5009918, 52.2805252, -220.5577393, 71.3639603, -233.8649597, 272.8382568
2: -86.8753586, 48.2073059, -117.3002396, 66.1966171, -153.0719757, 165.5075226
3: -112.7218323, 38.6586380, -153.0097809, 52.9897652, -165.7115631, 191.6683960
4: -61.1049271, 41.0587082, -82.8999481, 56.8029289, -117.9078522, 123.9586563

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239105, upper bound: 42.5232180
time: 0.89 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -23.6530495, 25.2891521, -47.5991364, 47.6846771
1: -175.0307159, 56.3754234, -183.5821686, 59.6651878, -234.6959076, 239.9575653
2: -93.4728622, 51.8776817, -98.9419250, 54.6604042, -148.1332550, 150.8195801
3: -121.6609573, 41.6777878, -127.8725586, 43.8773117, -165.5382690, 169.5503387
4: -65.8667603, 44.1967926, -69.8602829, 46.5234184, -112.3901825, 114.0570679

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5242959
time: 0.80 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239105, upper bound: 42.5232166
time: 0.83 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -28.7457638, 30.8050442, -53.1150246, 52.7773895
1: -175.0307159, 56.3754234, -222.4906006, 72.8255463, -247.8562469, 278.8660278
2: -93.4728622, 51.8776817, -119.8959045, 66.8507690, -160.3236389, 171.7735901
3: -121.6609573, 41.6777878, -155.2182922, 53.5666428, -175.2276001, 196.8960724
4: -65.8667603, 44.1967926, -84.7777328, 57.0106125, -122.8773422, 128.9745178

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244725, upper bound: 42.5242959
time: 0.86 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239105, upper bound: 42.5232166
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -25.7358856, 27.8050156, -50.1150017, 49.7675133
1: -175.0307159, 56.3754234, -201.8453674, 65.0860062, -240.1167297, 258.2207642
2: -93.4728622, 51.8776817, -107.5397186, 60.0969658, -153.5698242, 159.4173737
3: -121.6609573, 41.6777878, -140.1542664, 48.1530762, -169.8140259, 181.8320465
4: -65.8667603, 44.1967926, -75.9749756, 51.2600746, -117.1268158, 120.1717606

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243963, upper bound: 42.5240483
time: 0.94 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236604, upper bound: 42.5238021
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -22.3099861, 24.0316277, -28.6109943, 31.0786285, -53.3886147, 52.6426239
1: -175.0307159, 56.3754234, -223.9671783, 72.3581238, -247.3888397, 280.3425903
2: -93.4728622, 51.8776817, -118.9563522, 67.1506119, -160.6234741, 170.8340302
3: -121.6609573, 41.6777878, -155.3005066, 53.7572403, -175.4181976, 196.9782867
4: -65.8667603, 44.1967926, -84.0604477, 57.6275024, -123.4942627, 128.2572174

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243963, upper bound: 42.5240483
time: 0.95 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236604, upper bound: 42.5238021
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -23.8953743, 25.5614204, -48.2193832, 48.1494370
1: -176.2945251, 57.0708656, -186.9830933, 60.3684921, -236.6629944, 244.0539398
2: -94.9132462, 52.2951469, -100.6089172, 55.3390503, -150.2522888, 152.9040527
3: -122.6806107, 42.0384216, -130.1380463, 44.3996086, -167.0802155, 172.1764679
4: -66.9144669, 44.5387917, -70.7848587, 46.9832764, -113.8977432, 115.3236389

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.85 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.84 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -26.3633499, 28.4440651, -51.1020279, 50.6174049
1: -176.2945251, 57.0708656, -206.4710541, 66.6278229, -242.9223480, 263.5419006
2: -94.9132462, 52.2951469, -110.4162445, 61.4447594, -156.3580017, 162.7113647
3: -122.6806107, 42.0384216, -143.3764648, 49.2761269, -171.9567108, 185.4148865
4: -66.9144669, 44.5387917, -77.7718964, 52.5385818, -119.4530487, 122.3106842

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243969, upper bound: 42.5240946
time: 0.86 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -26.2780914, 28.3910427, -51.0489960, 50.5321541
1: -176.2945251, 57.0708656, -207.1456909, 66.5400772, -242.8345947, 264.2165222
2: -94.9132462, 52.2951469, -110.2955780, 61.4593582, -156.3726044, 162.5906982
3: -122.6806107, 42.0384216, -143.7563019, 49.2168350, -171.8973999, 185.7947235
4: -66.9144669, 44.5387917, -77.7333374, 52.3282928, -119.2427597, 122.2721252

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247183, upper bound: 42.5245417
time: 0.92 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239074, upper bound: 42.5240948
time: 0.92 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245908, upper bound: 42.5244425
time: 0.99 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -22.6579609, 24.2540627, -28.6043892, 31.1422596, -53.8002205, 52.8584518
1: -176.2945251, 57.0708656, -225.7873688, 72.4520645, -248.7465820, 282.8581543
2: -94.9132462, 52.2951469, -119.6162796, 67.2707214, -162.1839600, 171.9114075
3: -122.6806107, 42.0384216, -156.3691406, 53.8756599, -176.5562439, 198.4075623
4: -66.9144669, 44.5387917, -84.3199921, 57.6347504, -124.5492096, 128.8587494

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239074, upper bound: 42.5240948
time: 0.92 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245908, upper bound: 42.5244425
time: 0.96 seconds

## BFS NS instance: NS_A1_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -23.5677090, 25.2059841, -23.6167068, 25.3224068, -48.8901138, 48.8226929
1: -184.3307343, 59.5229073, -183.9517670, 59.5331726, -243.8639069, 243.4746704
2: -99.1971664, 54.5543060, -98.8578873, 54.6367569, -153.8339081, 153.4122009
3: -128.2790833, 43.7834854, -127.9907227, 43.8855515, -172.1646118, 171.7742004
4: -69.7876740, 46.3303642, -69.7239761, 46.5521240, -116.3397980, 116.0543213

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -23.5677090, 25.2059841, -26.3813496, 28.4505577, -52.0182648, 51.5873337
1: -184.3307343, 59.5229073, -204.9241791, 66.5419083, -250.8726044, 264.4470825
2: -99.1971664, 54.5543060, -109.6991882, 61.3693161, -160.5664673, 164.2534943
3: -128.2790833, 43.7834854, -142.3691101, 49.2498550, -177.5289307, 186.1525879
4: -69.7876740, 46.3303642, -77.5156021, 52.6544647, -122.4421387, 123.8459473

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235856, upper bound: 42.5239249
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -23.0098019, 24.6616135, -23.6167068, 25.3224068, -48.3322067, 48.2783203
1: -179.2008820, 57.9912300, -183.9517670, 59.5331726, -238.7340393, 241.9429626
2: -96.3431702, 53.1889877, -98.8578873, 54.6367569, -150.9799194, 152.0468750
3: -124.6888733, 42.7408600, -127.9907227, 43.8855515, -168.5743713, 170.7315674
4: -67.9345169, 45.3148346, -69.7239761, 46.5521240, -114.4866409, 115.0388107

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 16

## BFS NS instance: NS_A1_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -23.0098019, 24.6616135, -26.3813496, 28.4505577, -51.4603539, 51.0429611
1: -179.2008820, 57.9912300, -204.9241791, 66.5419083, -245.7427521, 262.9154053
2: -96.3431702, 53.1889877, -109.6991882, 61.3693161, -157.7124939, 162.8881683
3: -124.6888733, 42.7408600, -142.3691101, 49.2498550, -173.9387207, 185.1099396
4: -67.9345169, 45.3148346, -77.5156021, 52.6544647, -120.5889816, 122.8304367

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241888, upper bound: 42.5243181
time: 0.85 seconds

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237329, upper bound: 42.5239825
time: 0.93 seconds

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -23.6508427, 25.6981697, -22.0815239, 23.8715057, -47.5223465, 47.7796898
1: -187.3264313, 59.9331474, -173.6061249, 55.8173485, -243.1437836, 233.5392761
2: -99.1052551, 55.4655304, -92.5020294, 51.4314804, -150.5366974, 147.9675598
3: -129.8612976, 44.4966087, -120.5883484, 41.3429527, -171.2042389, 165.0849609
4: -69.8584442, 47.2790184, -65.1844482, 43.8699837, -113.7284241, 112.4634705

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249144, upper bound: 42.5248513
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245641, upper bound: 42.5248581
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -23.9437981, 26.0091267, -22.6793537, 24.4340954, -48.3778915, 48.6884804
1: -189.5855560, 60.6800919, -177.5234528, 57.2012672, -246.7867889, 238.2035522
2: -100.3765869, 56.1372452, -94.9949722, 52.6250420, -153.0016327, 151.1322174
3: -131.4502869, 45.0423508, -123.3468475, 42.3091774, -173.7594452, 168.3891907
4: -70.7577896, 47.8496780, -66.9352951, 44.9271812, -115.6849594, 114.7849731

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249273, upper bound: 42.5248620
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248703, upper bound: 42.5248847
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -28.5052929, 31.2166424, -21.4572887, 23.1450634, -51.6503563, 52.6739311
1: -220.9524536, 71.9431763, -168.1677704, 54.2197571, -275.1722107, 240.1109467
2: -116.6637421, 67.1353149, -89.7769241, 49.8931923, -166.5569305, 156.9122314
3: -153.0885773, 53.8016396, -116.9175491, 40.1043587, -193.1929169, 170.7191620
4: -82.9357376, 58.0095062, -63.3232803, 42.5446777, -125.4804153, 121.3327637

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245653, upper bound: 42.5240846
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236522, upper bound: 42.5239136
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -28.6221066, 31.3376808, -25.2288742, 27.1391163, -55.7612228, 56.5665474
1: -221.7456665, 72.1813431, -196.3530731, 64.1860275, -285.9317017, 268.5344238
2: -117.0034027, 67.4526520, -105.4150238, 59.0122566, -176.0156555, 172.8676300
3: -153.5200500, 54.0847702, -137.1155243, 47.2274895, -200.7475433, 191.2002869
4: -83.1782150, 58.2959709, -74.7429733, 50.1702118, -133.3484192, 133.0389404

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247382, upper bound: 42.5247256
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246988, upper bound: 42.5245936
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -28.5052929, 31.2166424, -22.3307762, 23.9714184, -52.4766998, 53.5474167
1: -220.9524536, 71.9431763, -173.6308594, 56.3216362, -277.2740784, 245.5740356
2: -116.6637421, 67.1353149, -93.3150711, 51.6865807, -168.3502808, 160.4503632
3: -153.0885773, 53.8016396, -120.8106766, 41.5318413, -194.6204224, 174.6123199
4: -82.9357376, 58.0095062, -65.8808517, 44.1220055, -127.0577393, 123.8903503

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245959, upper bound: 42.5247053
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245514, upper bound: 42.5245757
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -28.6221066, 31.3376808, -23.6066246, 25.5701199, -54.1922264, 54.9443054
1: -221.7456665, 72.1813431, -185.0536194, 59.6441536, -281.3898315, 257.2349548
2: -117.0034027, 67.4526520, -98.7579346, 55.0714645, -172.0748444, 166.2105560
3: -153.5200500, 54.0847702, -128.4872437, 44.2028198, -197.7228699, 182.5720215
4: -83.1782150, 58.2959709, -69.6707687, 47.1290512, -130.3072510, 127.9667358

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247359, upper bound: 42.5241301
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236567, upper bound: 42.5240642
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -24.3631954, 26.3139744, -22.2010403, 23.9670734, -48.3302689, 48.5150146
1: -191.2166901, 61.5759010, -176.2761993, 56.1774368, -247.3941193, 237.8520966
2: -101.8605347, 56.8321877, -93.8528748, 51.7352066, -153.5957336, 150.6850586
3: -132.6836090, 45.5869598, -122.2529984, 41.5449371, -174.2285461, 167.8399658
4: -71.9175568, 48.4946480, -65.8136597, 43.9355850, -115.8531418, 114.3083038

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249646, upper bound: 42.5248628
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249646, upper bound: 42.5248628
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -24.8362522, 26.8520069, -21.6534882, 23.3983707, -48.2346230, 48.5054932
1: -195.0066223, 62.7883835, -170.4662933, 54.7559242, -249.7625275, 233.2546692
2: -103.7632599, 58.0026894, -90.8322678, 50.4153099, -154.1785736, 148.8349609
3: -135.3092499, 46.5088081, -118.4171906, 40.5358963, -175.8451385, 164.9259949
4: -73.2668762, 49.5195808, -63.9834480, 42.9548721, -116.2217484, 113.5030136

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245096, upper bound: 42.5244067
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246662, upper bound: 42.5244973
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -24.5728741, 26.5497131, -22.9923649, 24.7495518, -49.3224258, 49.5420761
1: -192.7191772, 62.1105003, -179.6977234, 57.9785271, -250.6977081, 241.8082123
2: -102.6259460, 57.3521996, -96.2843094, 53.2970352, -155.9229584, 153.6365051
3: -133.7522583, 45.9915695, -124.9043350, 42.8572655, -176.6095276, 170.8959045
4: -72.4731827, 48.9668388, -67.8521957, 45.4986076, -117.9717636, 116.8190308

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241675, upper bound: 42.5239750
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238332, upper bound: 42.5238240
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -29.0785809, 31.8181019, -22.2142467, 23.9859886, -53.0645599, 54.0323410
1: -225.1883545, 73.4981689, -174.5162201, 56.1100540, -281.2983398, 248.0143890
2: -119.3765259, 68.4823990, -93.2432404, 51.6199837, -170.9965057, 161.7256470
3: -156.0846100, 54.8241577, -121.1633530, 41.5354919, -197.6201019, 175.9875183
4: -84.8334122, 59.1474266, -65.6411133, 44.0743942, -128.9078064, 124.7885437

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5029582, upper bound: 42.5028309
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238831, upper bound: 42.5235494
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -28.9306583, 31.6011829, -23.6276894, 25.3984032, -54.3290596, 55.2288704
1: -223.3292542, 73.1332550, -182.7615356, 59.5419769, -282.8712158, 255.8947754
2: -118.6273117, 68.1167526, -98.2261505, 54.8135872, -173.4408722, 166.3428955
3: -154.9183655, 54.4647331, -127.1330185, 43.9667816, -198.8851318, 181.5977478
4: -84.3526688, 58.8210030, -69.4097137, 46.9460793, -131.2987366, 128.2307129

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5230692, upper bound: 42.5215200
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5205430, upper bound: 42.5204723
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.38 seconds
NS_A1_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5240352, upper bound: 42.5234643
NS_A1_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
NS_A1_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5239105, upper bound: 42.5232180
NS_A1_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
NS_A1_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5243963, upper bound: 42.5240483
NS_A1_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5236604, upper bound: 42.5238021
NS_A1_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5243963, upper bound: 42.5240483
NS_A1_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5236604, upper bound: 42.5238021
NS_A1_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5243969, upper bound: 42.5240946
NS_A1_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5244089, upper bound: 42.5240946
NS_A1_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5239074, upper bound: 42.5240948
NS_A1_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245908, upper bound: 42.5244425
NS_A1_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5239074, upper bound: 42.5240948
NS_A1_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245908, upper bound: 42.5244425
NS_A2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5249144, upper bound: 42.5248513
NS_A2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245641, upper bound: 42.5248581
NS_A2_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5249273, upper bound: 42.5248620
NS_A2_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5248703, upper bound: 42.5248847
NS_A2_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245653, upper bound: 42.5240846
NS_A2_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5236522, upper bound: 42.5239136
NS_A2_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5247382, upper bound: 42.5247256
NS_A2_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5246988, upper bound: 42.5245936
NS_A2_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245959, upper bound: 42.5247053
NS_A2_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245514, upper bound: 42.5245757
NS_A2_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5247359, upper bound: 42.5241301
NS_A2_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5236567, upper bound: 42.5240642
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5249646, upper bound: 42.5248628
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5249646, upper bound: 42.5248628
NS_A2_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5245096, upper bound: 42.5244067
NS_A2_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5246662, upper bound: 42.5244973
NS_A2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5241675, upper bound: 42.5239750
NS_A2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5238332, upper bound: 42.5238240
NS_A2_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5029582, upper bound: 42.5028309
NS_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5238831, upper bound: 42.5235494
NS_A2_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5230692, upper bound: 42.5215200
NS_A2_B1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.38
Output dim: 0, lower bound: -42.5205430, upper bound: 42.5204723

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -20.4780102, 21.9703808, -24.0813408, 26.1088219, -46.5868301, 46.0517159
1: -160.3094482, 51.5184288, -189.9620972, 60.9411125, -221.2505341, 241.4804840
2: -85.7003708, 47.4727249, -100.9842300, 56.3064842, -142.0068359, 148.4569397
3: -111.1902542, 38.0833168, -131.8000336, 45.1754990, -156.3657532, 169.8833313
4: -60.2077942, 40.4373589, -71.2031479, 48.0144501, -108.2222290, 111.6404877

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241652, upper bound: 42.5243692
time: 0.85 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241652, upper bound: 42.5243692
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -20.4383183, 21.9645157, -24.6454945, 26.6351318, -47.0734482, 46.6100006
1: -160.0944214, 51.4324913, -193.0879974, 62.3455467, -222.4399567, 244.5204926
2: -85.4360733, 47.4517136, -102.9717941, 57.5210075, -142.9570770, 150.4235077
3: -111.0303955, 38.0737534, -134.1105042, 46.1014977, -157.1318817, 172.1842651
4: -60.1185684, 40.4412346, -72.8011475, 49.0771866, -109.1957550, 113.2423782

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241652, upper bound: 42.5243692
time: 0.88 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235290, upper bound: 42.5240625
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -20.4780102, 21.9703808, -26.9745388, 29.4028416, -49.8808517, 48.9449158
1: -160.3094482, 51.5184288, -211.9806061, 68.2333832, -228.5428314, 263.4990234
2: -85.7003708, 47.4727249, -112.3465118, 63.3932457, -149.0936127, 159.8192444
3: -111.1902542, 38.0833168, -146.8511963, 50.8077278, -161.9979858, 184.9344940
4: -60.2077942, 40.4373589, -79.2908478, 54.4418564, -114.6496201, 119.7282104

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -20.4383183, 21.9645157, -27.5763893, 29.9701519, -50.4084702, 49.5409050
1: -160.0944214, 51.4324913, -215.5299377, 69.7355957, -229.8300171, 266.9624329
2: -85.4360733, 47.4517136, -114.5864944, 64.6936798, -150.1297455, 162.0381927
3: -111.0303955, 38.0737534, -149.4986877, 51.8054085, -162.8357849, 187.5724030
4: -60.1185684, 40.4412346, -81.0149002, 55.5548592, -115.6734085, 121.4561310

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
time: 0.81 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5229604
time: 0.85 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -21.9223747, 23.6351528, -24.4832306, 26.5806713, -48.5030441, 48.1183815
1: -171.9929047, 55.3753586, -193.5201416, 61.9644852, -233.9573822, 248.8954773
2: -91.7921753, 50.9705620, -102.7222672, 57.3130760, -149.1052399, 153.6928253
3: -119.5258026, 40.9590950, -134.1976166, 45.9820251, -165.5078278, 175.1567078
4: -64.6928940, 43.4558525, -72.3936234, 48.8807564, -113.5736542, 115.8494720

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236493, upper bound: 42.5222966
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245612, upper bound: 42.5243721
time: 0.85 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -21.9901352, 23.7093735, -25.0863342, 27.1436367, -49.1337624, 48.7957077
1: -172.6207428, 55.5729027, -196.8486786, 63.4622269, -236.0829620, 252.4215851
2: -92.1502838, 51.1444435, -104.8239670, 58.6036148, -150.7538605, 155.9684143
3: -119.9719849, 41.1013832, -136.6574707, 46.9740105, -166.9459991, 177.7588501
4: -64.9461746, 43.5906525, -74.0947189, 50.0207253, -114.9669037, 117.6853638

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2_A1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242649, upper bound: 42.5243722
time: 0.98 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2_A2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235296, upper bound: 42.5243722
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -21.9223747, 23.6351528, -27.3444958, 29.8449554, -51.7673302, 50.9796448
1: -171.9929047, 55.3753586, -215.5350037, 69.2036819, -241.1965942, 270.9103394
2: -91.7921753, 50.9705620, -114.0214615, 64.3390656, -156.1312103, 164.9920197
3: -119.5258026, 40.9590950, -149.2216034, 51.5704155, -171.0962067, 190.1806793
4: -64.6928940, 43.4558525, -80.4342499, 55.2441902, -119.9370880, 123.8900986

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5238021
time: 0.82 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236604, upper bound: 42.5238021
time: 0.86 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -21.9901352, 23.7093735, -27.9589577, 30.4124527, -52.4025764, 51.6683311
1: -172.6207428, 55.5729027, -218.9195862, 70.7225647, -243.3433075, 274.4924622
2: -92.1502838, 51.1444435, -116.2368698, 65.6436234, -157.7938538, 167.3813171
3: -119.9719849, 41.1013832, -151.7872925, 52.5696144, -172.5415955, 192.8886719
4: -64.9461746, 43.5906525, -82.1681137, 56.3753738, -121.3215485, 125.7587662

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236604, upper bound: 42.5238021
time: 0.81 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228780, upper bound: 42.5238021
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -23.5677090, 25.2059841, -23.8953743, 25.5614204, -49.1291275, 49.1013565
1: -184.3307343, 59.5229073, -186.9830933, 60.3684921, -244.6991882, 246.5059967
2: -99.1971664, 54.5543060, -100.6089172, 55.3390503, -154.5361938, 155.1632233
3: -128.2790833, 43.7834854, -130.1380463, 44.3996086, -172.6786957, 173.9215393
4: -69.7876740, 46.3303642, -70.7848587, 46.9832764, -116.7709503, 117.1152039

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5233172, upper bound: 42.5239275
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5231265, upper bound: 42.5230718
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -23.0098019, 24.6616135, -23.8953743, 25.5614204, -48.5712204, 48.5569878
1: -179.2008820, 57.9912300, -186.9830933, 60.3684921, -239.5693359, 244.9743042
2: -96.3431702, 53.1889877, -100.6089172, 55.3390503, -151.6822052, 153.7978973
3: -124.6888733, 42.7408600, -130.1380463, 44.3996086, -169.0884705, 172.8789062
4: -67.9345169, 45.3148346, -70.7848587, 46.9832764, -114.9177933, 116.0996933

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239497, upper bound: 42.5232274
time: 0.71 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5231265, upper bound: 42.5230718
time: 1.58 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -23.5677090, 25.2059841, -26.3633499, 28.4440651, -52.0117722, 51.5693321
1: -184.3307343, 59.5229073, -206.4710541, 66.6278229, -250.9585419, 265.9939575
2: -99.1971664, 54.5543060, -110.4162445, 61.4447594, -160.6419220, 164.9705353
3: -128.2790833, 43.7834854, -143.3764648, 49.2761269, -177.5551910, 187.1599426
4: -69.7876740, 46.3303642, -77.7718964, 52.5385818, -122.3262482, 124.1022568

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236273, upper bound: 42.5222997
time: 0.84 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -23.0098019, 24.6616135, -26.3633499, 28.4440651, -51.4538651, 51.0249596
1: -179.2008820, 57.9912300, -206.4710541, 66.6278229, -245.8286896, 264.4622498
2: -96.3431702, 53.1889877, -110.4162445, 61.4447594, -157.7879333, 163.6051941
3: -124.6888733, 42.7408600, -143.3764648, 49.2761269, -173.9649506, 186.1172943
4: -67.9345169, 45.3148346, -77.7718964, 52.5385818, -120.4730988, 123.0867310

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236273, upper bound: 42.5222997
time: 0.78 seconds

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A1_A2_B1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -21.9347095, 23.4706821, -24.9733219, 26.9887924, -48.9234962, 48.4440041
1: -170.9280548, 55.2667961, -197.6780396, 63.3079453, -234.2359924, 252.9448090
2: -92.0328903, 50.6182175, -105.1675415, 58.4621468, -150.4950409, 155.7857208
3: -118.9630051, 40.6904068, -137.1755676, 46.8101196, -165.7730865, 177.8659668
4: -64.8530426, 43.0747604, -74.0445709, 49.7030754, -114.5561218, 117.1193314

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5231671, upper bound: 42.5219690
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5194437, upper bound: 42.5196270
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -21.7738171, 23.3118248, -25.2165070, 27.2878704, -49.0616875, 48.5283318
1: -169.9152527, 54.8735809, -200.2385406, 63.9675407, -233.8827972, 255.1121216
2: -91.4528351, 50.2606201, -106.3586655, 59.1054459, -150.5582581, 156.6192932
3: -118.2460327, 40.4127541, -138.8876190, 47.3354034, -165.5814362, 179.3003387
4: -64.3886490, 42.7499542, -74.7981720, 50.2109337, -114.5995789, 117.5481262

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2_A1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5239468, upper bound: 42.5241482
time: 0.95 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B1_B2_A2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245681, upper bound: 42.5245655
time: 0.83 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -21.9347095, 23.4706821, -27.2192459, 29.6474934, -51.5821991, 50.6899261
1: -170.9280548, 55.2667961, -215.7488403, 69.0250854, -239.9531403, 271.0156250
2: -92.0328903, 50.6182175, -114.1871643, 64.0831604, -156.1160583, 164.8053589
3: -118.9630051, 40.6904068, -149.4018097, 51.3186226, -170.2816010, 190.0922241
4: -64.8530426, 43.0747604, -80.4098892, 54.8295975, -119.6826324, 123.4846344

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5229781, upper bound: 42.5217414
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238759, upper bound: 42.5240842
time: 0.84 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238759, upper bound: 42.5240948
time: 0.85 seconds

## BFS NS instance: NS_A1_A1_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -21.7738171, 23.3118248, -27.6545792, 30.1724472, -51.9462662, 50.9664040
1: -169.9152527, 54.8735809, -220.2975464, 70.2003784, -240.1155853, 275.1710510
2: -91.4528351, 50.2606201, -116.2928848, 65.2093811, -156.6621857, 166.5534973
3: -118.2460327, 40.4127541, -152.4376373, 52.2483292, -170.4943542, 192.8503876
4: -64.3886490, 42.7499542, -81.7676239, 55.7350998, -120.1237335, 124.5175781

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240746, upper bound: 42.5232224
time: 0.87 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5232310, upper bound: 42.5229431
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -22.9906063, 24.9743938, -22.8168640, 24.5310421, -47.5216446, 47.7912560
1: -182.3299255, 58.2847214, -180.0440063, 57.8053513, -240.1352692, 238.3287201
2: -96.5021286, 53.9043579, -96.2600327, 53.1754684, -149.6775970, 150.1643829
3: -126.3973541, 43.2493896, -125.1644745, 42.6105728, -169.0079346, 168.4138641
4: -68.0093536, 45.9029160, -67.6462250, 45.0976295, -113.1069794, 113.5491409

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246129, upper bound: 42.5244609
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246129, upper bound: 42.5248299
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -23.4655628, 25.4930439, -21.6832428, 23.4280987, -46.8936615, 47.1762772
1: -185.9250946, 59.4722443, -170.6664276, 54.8329201, -240.7580109, 230.1386719
2: -98.3748322, 55.0226746, -90.9530106, 50.4836807, -148.8585052, 145.9756775
3: -128.8974762, 44.1459618, -118.5636520, 40.5889168, -169.4863892, 162.7096100
4: -69.3385239, 46.8853683, -64.0731506, 43.0119629, -112.3504868, 110.9585037

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242282, upper bound: 42.5243452
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244283, upper bound: 42.5244283
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -23.3030396, 25.3002682, -22.8680820, 24.5737305, -47.8767700, 48.1683502
1: -184.6858978, 59.0749283, -179.7523041, 57.8179550, -242.5038452, 238.8272400
2: -97.8349304, 54.6070480, -96.3182220, 53.1248283, -150.9597626, 150.9252625
3: -128.0604553, 43.8220863, -124.9202652, 42.6556816, -170.7161407, 168.7423401
4: -68.9547577, 46.4986458, -67.7579193, 45.1775169, -114.1322479, 114.2565613

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5107501, upper bound: 42.5143728
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245895, upper bound: 42.5243521
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243139, upper bound: 42.5243133
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -23.7567654, 25.8028755, -22.2666225, 23.9817963, -47.7385635, 48.0694962
1: -188.1803284, 60.2151184, -174.4701996, 56.1796532, -244.3599854, 234.6853180
2: -99.6406631, 55.6918640, -93.3826447, 51.6574097, -151.2980652, 149.0744781
3: -130.4826660, 44.6894798, -121.2387314, 41.5325508, -172.0151672, 165.9282074
4: -70.2334595, 47.4538460, -65.7870483, 44.0613899, -114.2948456, 113.2408905

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244481, upper bound: 42.5246251
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5248703, upper bound: 42.5248847
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -28.3241043, 31.0170059, -20.9107208, 22.5381241, -50.8622246, 51.9277267
1: -219.5289001, 71.4935608, -163.9398041, 52.8418808, -272.3707581, 235.4333649
2: -115.9427719, 66.7015839, -87.6047211, 48.5861511, -164.5288849, 154.3063049
3: -152.1203308, 53.4450493, -114.0287247, 39.0531311, -191.1734619, 167.4737701
4: -82.4278488, 57.6235619, -61.7714806, 41.3789177, -123.8067627, 119.3950424

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236522, upper bound: 42.5239136
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236522, upper bound: 42.5239136
time: 0.82 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.11 + 418.04 = 421.15 seconds
