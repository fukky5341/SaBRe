## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 495.41142893616905


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-93.3107758, 99.4533081, -93.3107758, 99.4533081, -192.7640686, 192.7640686)
1: (-688.5435181, 232.2499542, -688.5435181, 232.2499542, -920.7934570, 920.7934570)
2: (-374.2228088, 214.8466492, -374.2228088, 214.8466492, -589.0694580, 589.0694580)
3: (-477.7779846, 172.3284760, -477.7779846, 172.3284760, -650.1064453, 650.1064453)
4: (-267.4408569, 186.4021912, -267.4408569, 186.4021912, -453.8430481, 453.8430481)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 2.40 = 3.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -495.4163831, upper bound: 495.4163831

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4161894, upper bound: 495.4160291
time: 0.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4160210, upper bound: 495.4160210
time: 1.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.16 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 3, lower bound: -495.4161894, upper bound: 495.4160291
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 3, lower bound: -495.4160210, upper bound: 495.4160210

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -91.1607666, 97.1320343, -91.9239578, 97.9515457, -189.1122894, 189.0559998
1: -672.9181519, 226.9340973, -678.4035034, 228.8189545, -901.7371216, 905.3375854
2: -365.8666382, 209.8396912, -368.8190308, 211.6081085, -577.4747314, 578.6586914
3: -467.0450134, 168.3161774, -470.8204346, 169.7328339, -636.7778320, 639.1365967
4: -261.3837280, 181.9539795, -263.5302124, 183.5269775, -444.9107056, 445.4841919

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4159233, upper bound: 495.4157480
time: 0.97 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4160600, upper bound: 495.4159040
time: 1.06 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -104.4515686, 110.4484024, -91.9413452, 97.9522095, -202.4037781, 202.3897400
1: -751.6062012, 259.6625061, -677.5537720, 228.7957153, -980.4019165, 937.2162476
2: -415.5630188, 238.4505920, -368.6148376, 211.5044250, -627.0674438, 607.0654297
3: -524.5878296, 191.3292389, -470.3003845, 169.6615448, -694.2493286, 661.6295166
4: -298.7754517, 207.6202240, -263.4842224, 183.4642181, -482.2396851, 471.1043701

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157220, upper bound: 495.4157247
time: 1.10 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4158957, upper bound: 495.4158957
time: 1.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.55 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -495.4159233, upper bound: 495.4157480
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -495.4160600, upper bound: 495.4159040
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -495.4157220, upper bound: 495.4157247
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -495.4158957, upper bound: 495.4158957

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -90.2451477, 96.1400299, -91.4804077, 97.4284286, -187.6735840, 187.6204224
1: -666.2156372, 224.6707916, -674.0670776, 227.5782776, -893.7938843, 898.7378540
2: -362.2077026, 207.7150574, -366.7163391, 210.3924866, -572.6002197, 574.4313965
3: -462.3937683, 166.6159973, -468.0419922, 168.7453156, -631.1390991, 634.6579590
4: -258.7729187, 180.0821991, -262.1275940, 182.4847717, -441.2576294, 442.2097778

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4145116
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145976, upper bound: 495.4144955
time: 1.02 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -89.9625549, 95.8509521, -90.1765747, 96.0833740, -186.0458984, 186.0275116
1: -663.9912109, 223.9255219, -665.3417358, 224.4273224, -888.4185181, 889.2672119
2: -360.9765625, 207.0569763, -361.6729126, 207.5491028, -568.5255737, 568.7298584
3: -460.7901001, 166.1037750, -461.6680908, 166.5045319, -627.2946167, 627.7718506
4: -257.8826599, 179.5605621, -258.4166870, 180.0404816, -437.9230957, 437.9772339

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149024, upper bound: 495.4149327
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145976, upper bound: 495.4149176
time: 1.11 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -103.6554108, 109.5787430, -91.5933914, 97.5431137, -201.1984711, 201.1720886
1: -745.7239990, 257.7000122, -674.0708008, 227.8030090, -973.5269775, 931.7708130
2: -412.3775940, 236.5954742, -366.9127502, 210.5336456, -622.9112549, 603.5082397
3: -520.5234985, 189.8426666, -468.0814514, 168.8679657, -689.3914185, 657.9241333
4: -296.5080261, 205.9808960, -262.3652954, 182.6393280, -479.1473389, 468.3461304

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144654, upper bound: 495.4145017
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143888, upper bound: 495.4143343
time: 1.07 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -103.2366180, 109.1459122, -90.2023773, 96.0979233, -199.3345184, 199.3482971
1: -742.4318848, 256.5968628, -664.5650024, 224.4208679, -966.8527832, 921.1618042
2: -410.5567932, 235.6236572, -361.4898071, 207.4653778, -618.0221558, 597.1134644
3: -518.1647339, 189.0790558, -461.1902466, 166.4465332, -684.6112671, 650.2692871
4: -295.2056580, 205.1992645, -258.3874207, 179.9995880, -475.2052307, 463.5866394

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4148284, upper bound: 495.4149188
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147231, upper bound: 495.4147231
time: 1.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.49 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4145116
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4145976, upper bound: 495.4144955
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4149024, upper bound: 495.4149327
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4145976, upper bound: 495.4149176
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4144654, upper bound: 495.4145017
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4143888, upper bound: 495.4143343
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4148284, upper bound: 495.4149188
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 3, lower bound: -495.4147231, upper bound: 495.4147231

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -87.4211197, 93.0748596, -89.6368256, 95.4357910, -182.8568878, 182.7116394
1: -645.2709351, 217.6314240, -660.5328979, 222.9780579, -868.2490234, 878.1642456
2: -350.7425842, 201.2241058, -359.2543335, 206.1725311, -556.9151001, 560.4783325
3: -447.8721313, 161.3867645, -458.6387329, 165.3442993, -613.2164307, 620.0255127
4: -250.4988098, 174.4324188, -256.7268677, 178.8069000, -429.3056335, 431.1592102

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4145116
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4145116
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -88.5076370, 94.2270126, -89.5031357, 95.3198166, -183.8274078, 183.7301483
1: -653.4435425, 220.4784851, -660.0250854, 222.7054138, -876.1488647, 880.5034790
2: -355.5723572, 203.6327362, -358.9166565, 205.8508606, -561.4232178, 562.5493774
3: -453.7387390, 163.3110962, -458.2510376, 165.1164246, -618.8551025, 621.5621338
4: -254.0021820, 176.4128876, -256.5249329, 178.4756012, -432.4777527, 432.9378052

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4144955
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4144955
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -87.1522827, 92.7972946, -88.2437973, 93.9910278, -181.1432953, 181.0410919
1: -643.1206665, 216.9231720, -651.0684814, 219.6100922, -862.7306519, 867.9916382
2: -349.5678711, 200.5936279, -353.8258972, 203.1226044, -552.6904907, 554.4195557
3: -446.3232117, 160.8994141, -451.7564697, 162.9392242, -609.2624512, 612.6558838
4: -249.6535339, 173.9317322, -252.7414398, 176.1910248, -425.8445435, 426.6731567

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149024, upper bound: 495.4149327
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149024, upper bound: 495.4149327
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -88.1497803, 93.8820343, -88.2237320, 94.0022812, -182.1520081, 182.1057587
1: -650.7974243, 219.5454102, -651.4637451, 219.6144867, -870.4119263, 871.0091553
2: -354.0279846, 202.8409729, -353.9752808, 203.0652008, -557.0930786, 556.8162231
3: -451.7862244, 162.6967163, -451.9922485, 162.9221497, -614.7083740, 614.6889648
4: -252.8812103, 175.7949829, -252.8780670, 176.0863800, -428.9675293, 428.6730347

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149237, upper bound: 495.4149176
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149237, upper bound: 495.4149176
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -100.5215836, 106.2577362, -89.9616776, 95.7918930, -196.3134155, 196.2194214
1: -723.5298462, 249.9182434, -662.3034058, 223.7359772, -947.2658081, 912.2215576
2: -399.8504028, 229.5321045, -360.3381348, 206.8234558, -606.6738281, 589.8701782
3: -505.0037231, 184.1423950, -459.8939209, 165.8711853, -670.8748169, 644.0363159
4: -287.3907776, 199.8298798, -257.5888672, 179.4030914, -466.7938843, 457.4187622

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142574, upper bound: 495.4144271
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142566, upper bound: 495.4143351
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -101.1994553, 106.8274994, -89.3109741, 95.0843811, -196.2838440, 196.1384735
1: -727.3493042, 251.7278137, -657.5775146, 222.1672821, -949.5166016, 909.3052979
2: -402.8656921, 230.7235107, -357.8750610, 205.2464752, -608.1119995, 588.5985718
3: -507.9755249, 185.1370239, -456.6324768, 164.6437531, -672.6191406, 641.7695312
4: -289.6880493, 200.6952820, -255.8834686, 177.9697876, -467.6578369, 456.5787354

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141525, upper bound: 495.4142365
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141449, upper bound: 495.4141329
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -100.1314011, 105.8528595, -88.5023880, 94.2639160, -194.3953247, 194.3552399
1: -720.4275513, 248.8920746, -652.2225342, 220.1967468, -940.6242676, 901.1145020
2: -398.1661072, 228.6228333, -354.6382751, 203.5939636, -601.7600708, 583.2611084
3: -502.7949524, 183.4295044, -452.6147461, 163.3187866, -666.1137085, 636.0442505
4: -286.1831970, 199.0971527, -253.4098206, 176.6209869, -462.8041992, 452.5069580

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142574, upper bound: 495.4147569
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145183, upper bound: 495.4146045
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -100.6827011, 106.3102264, -87.9733734, 93.6973038, -194.3800049, 194.2835999
1: -723.3619385, 250.3776703, -648.4418335, 218.9161987, -942.2781372, 898.8193359
2: -400.6200256, 229.5579071, -352.6698303, 202.3036957, -602.9237061, 582.2277222
3: -505.0975647, 184.2203217, -450.0032349, 162.3209839, -667.4185791, 634.2235718
4: -288.0802002, 199.7720642, -252.0556183, 175.4453125, -463.5255127, 451.8275757

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141525, upper bound: 495.4145340
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143765, upper bound: 495.4143765
time: 1.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.11 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4145116
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4145116
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4144955
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4145333, upper bound: 495.4144955
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4149024, upper bound: 495.4149327
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4149024, upper bound: 495.4149327
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4149237, upper bound: 495.4149176
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4149237, upper bound: 495.4149176
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4142574, upper bound: 495.4144271
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4142566, upper bound: 495.4143351
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4141525, upper bound: 495.4142365
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4141449, upper bound: 495.4141329
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4142574, upper bound: 495.4147569
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4145183, upper bound: 495.4146045
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4141525, upper bound: 495.4145340
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -495.4143765, upper bound: 495.4143765

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -87.4211197, 93.0748596, -88.7875977, 94.5157776, -181.9368896, 181.8624115
1: -645.2709351, 217.6314240, -654.3173218, 220.8754425, -866.1463623, 871.9486694
2: -350.7425842, 201.2241058, -355.9349670, 204.1835785, -554.9261475, 557.1589355
3: -447.8721313, 161.3867645, -454.3690491, 163.7565155, -611.6286621, 615.7557983
4: -250.4988098, 174.4324188, -254.3248444, 177.0414276, -427.5402222, 428.7572632

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144463, upper bound: 495.4143582
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -87.4211197, 93.0748596, -101.8182373, 107.6477432, -195.0688629, 194.8930511
1: -645.2709351, 217.6314240, -732.6353760, 253.0567780, -898.3276367, 950.2667847
2: -350.7425842, 201.2241058, -405.0135498, 232.3882141, -583.1307983, 606.2376709
3: -447.8721313, 161.3867645, -511.5222168, 186.4331665, -634.3052979, 672.9089966
4: -250.4988098, 174.4324188, -291.1712952, 202.2967377, -452.7955322, 465.6036987

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144463, upper bound: 495.4143582
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143448
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -88.5076370, 94.2270126, -88.7465668, 94.5056458, -183.0132751, 182.9735718
1: -653.4435425, 220.4784851, -654.5772095, 220.8359222, -874.2793579, 875.0556641
2: -355.5723572, 203.6327362, -355.9837341, 204.0936127, -559.6659546, 559.6164551
3: -453.7387390, 163.3110962, -454.4991150, 163.7107086, -617.4494629, 617.8101807
4: -254.0021820, 176.4128876, -254.3950043, 176.9121246, -430.9143066, 430.8078308

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145018, upper bound: 495.4143416
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143292
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -88.5076370, 94.2270126, -102.1876755, 107.9732437, -196.4808502, 196.4146881
1: -653.4435425, 220.4784851, -734.5470581, 253.9809113, -907.4244385, 955.0255127
2: -355.5723572, 203.6327362, -406.4461060, 233.0521088, -588.6244507, 610.0788574
3: -453.7387390, 163.3110962, -512.9622192, 187.0130615, -640.7517090, 676.2733154
4: -254.0021820, 176.4128876, -292.3089905, 202.8321991, -456.8343811, 468.7218628

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145018, upper bound: 495.4143416
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -87.1522827, 92.7972946, -87.4035263, 93.0803986, -180.2326813, 180.2008209
1: -643.1206665, 216.9231720, -644.9586792, 217.5354919, -860.6561279, 861.8817749
2: -349.5678711, 200.5936279, -350.5608521, 201.1611481, -550.7290039, 551.1544800
3: -446.3232117, 160.8994141, -447.5587463, 161.3708191, -607.6939697, 608.4581299
4: -249.6535339, 173.9317322, -250.3771362, 174.4452667, -424.0987854, 424.3088684

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147379, upper bound: 495.4146568
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145652, upper bound: 495.4146393
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -87.1522827, 92.7972946, -100.5580521, 106.2984390, -193.4506836, 193.3553162
1: -643.1206665, 216.9231720, -723.1603394, 249.9205322, -893.0411987, 940.0834351
2: -349.5678711, 200.5936279, -399.7921448, 229.5383759, -579.1062622, 600.3857422
3: -446.3232117, 160.8994141, -504.7045288, 184.1820068, -630.5052490, 665.6039429
4: -249.6535339, 173.9317322, -287.4075317, 199.9149323, -449.5684814, 461.3392639

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147379, upper bound: 495.4146568
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145651, upper bound: 495.4146393
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -88.1497803, 93.8820343, -87.4924545, 93.2152481, -181.3650208, 181.3744354
1: -650.7974243, 219.5454102, -646.2172241, 217.8099670, -868.6074219, 865.7626343
2: -354.0279846, 202.8409729, -351.1501770, 201.3676147, -555.3956299, 553.9911499
3: -451.7862244, 162.6967163, -448.3830566, 161.5637054, -613.3499146, 611.0797729
4: -252.8812103, 175.7949829, -250.8279266, 174.5733490, -427.4544678, 426.6228638

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147581, upper bound: 495.4146364
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145652, upper bound: 495.4146197
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -88.1497803, 93.8820343, -100.9560471, 106.6619568, -194.8117218, 194.8380585
1: -650.7974243, 219.5454102, -725.3019409, 250.8935852, -901.6909790, 944.8473511
2: -354.0279846, 202.8409729, -401.2908325, 230.2783966, -584.3062744, 604.1317139
3: -451.7862244, 162.6967163, -506.2770691, 184.8160095, -636.6022339, 668.9737549
4: -252.8812103, 175.7949829, -288.5966492, 200.5317535, -453.4129639, 464.3916016

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147581, upper bound: 495.4146364
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4145879, upper bound: 495.4146197
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -98.7278366, 104.4594803, -85.9706726, 91.7726364, -190.5004425, 190.4301453
1: -710.8190308, 245.3277588, -633.6813965, 213.5036774, -924.3226929, 879.0091553
2: -392.4835815, 225.5461426, -343.7945557, 197.9020233, -590.3856201, 569.3406982
3: -495.7947083, 180.9664459, -439.1304626, 158.7438812, -654.5384521, 620.0969238
4: -282.0303955, 196.5077972, -245.5957489, 172.0588379, -454.0892334, 442.1035461

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4142512
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142033, upper bound: 495.4143539
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -100.4762115, 106.2057037, -94.7126617, 100.7800140, -201.2561951, 200.9183655
1: -723.1516724, 249.8031311, -696.3213501, 235.6712494, -958.8229370, 946.1243286
2: -399.6609192, 229.4218750, -379.3615112, 217.6616364, -617.3225098, 608.7833862
3: -504.7459106, 184.0540161, -483.7046204, 174.5414886, -679.2874146, 667.7586060
4: -287.2585144, 199.7337952, -271.3280029, 188.7520447, -476.0105591, 471.0617981

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4141785
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142021, upper bound: 495.4142661
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -99.3821869, 105.0030212, -85.3138351, 91.0623627, -190.4445343, 190.3168640
1: -714.4490356, 247.0774689, -628.9613647, 211.9169006, -926.3659058, 876.0387573
2: -395.3924866, 226.6830902, -341.2991943, 196.3093109, -591.7017822, 567.9822998
3: -498.6379089, 181.9094849, -435.8636169, 157.5048676, -656.1426392, 617.7730713
4: -284.2568359, 197.3286591, -243.8679047, 170.6110840, -454.8679199, 441.1965637

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141501
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141490
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -101.1638107, 106.7860336, -94.0581741, 100.0746155, -201.2384338, 200.8442078
1: -727.0581665, 251.6381073, -691.7255249, 234.0989075, -961.1569214, 943.3636475
2: -402.7194214, 230.6371155, -376.9234009, 216.0820312, -618.8014526, 607.5605469
3: -507.7774353, 185.0676117, -480.5140686, 173.3195496, -681.0968628, 665.5816650
4: -289.5857849, 200.6188354, -269.5968933, 187.3065643, -476.8923340, 470.2157288

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140782, upper bound: 495.4140522
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140777, upper bound: 495.4140472
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -98.3489151, 104.0669479, -84.4971085, 90.2502213, -188.5991364, 188.5640106
1: -707.8043213, 244.3269806, -623.9177246, 209.9531555, -917.7574463, 868.2446899
2: -390.8479614, 224.6640625, -338.2071228, 194.6770782, -585.5250244, 562.8711548
3: -493.6445312, 180.2740326, -432.0548401, 156.2133636, -649.8579102, 612.3288574
4: -280.8559875, 195.8007965, -241.4616394, 169.2203522, -450.0762939, 437.2624512

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144519, upper bound: 495.4146794
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144544, upper bound: 495.4147169
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -100.0712204, 105.7836151, -92.0858536, 98.1721039, -198.2433014, 197.8694763
1: -719.9280396, 248.7401886, -679.3124390, 229.2322540, -949.1602783, 928.0526123
2: -397.9136353, 228.4780426, -369.0360413, 211.9892883, -609.9027710, 597.5140991
3: -502.4537659, 183.3130951, -471.2131653, 170.0731201, -672.5268555, 654.5262451
4: -286.0073547, 198.9709930, -263.6978455, 183.8765717, -469.8839111, 462.6688232

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144180, upper bound: 495.4145134
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144200, upper bound: 495.4145562
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -98.8701630, 104.4908752, -83.9587021, 89.6778870, -188.5480499, 188.4495850
1: -710.5018311, 245.7373810, -620.1043701, 208.6467285, -919.1484985, 865.8417358
2: -393.1683350, 225.5298615, -336.2009888, 193.3668976, -586.5352173, 561.7308350
3: -495.7799377, 181.0056915, -429.4086609, 155.1988678, -650.9788208, 610.4143677
4: -282.6591492, 196.4245300, -240.0724640, 168.0268250, -450.6859741, 436.4969482

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143432, upper bound: 495.4145120
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143414, upper bound: 495.4144682
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -100.6318436, 106.2516403, -91.5544434, 97.6075668, -198.2393494, 197.8060913
1: -722.9515991, 250.2500305, -675.5893555, 227.9510040, -950.9025879, 925.8393555
2: -400.4109497, 229.4361420, -367.0706787, 210.6976166, -611.1085815, 596.5068359
3: -504.8172913, 184.1222839, -468.6140137, 169.0808868, -673.8980713, 652.7362671
4: -287.9337769, 199.6648254, -262.3115540, 182.6978760, -470.6316528, 461.9763794

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4136404, upper bound: 495.4136051
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4135069, upper bound: 495.4135069
time: 0.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.59 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4144463, upper bound: 495.4143582
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4144463, upper bound: 495.4143582
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143448
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4145018, upper bound: 495.4143416
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143292
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4145018, upper bound: 495.4143416
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4147379, upper bound: 495.4146568
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4145652, upper bound: 495.4146393
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4147379, upper bound: 495.4146568
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4145651, upper bound: 495.4146393
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4147581, upper bound: 495.4146364
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4145652, upper bound: 495.4146197
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4147581, upper bound: 495.4146364
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4145879, upper bound: 495.4146197
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4142512
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4142033, upper bound: 495.4143539
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4141785
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4142021, upper bound: 495.4142661
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141501
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141490
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4140782, upper bound: 495.4140522
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4140777, upper bound: 495.4140472
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4144519, upper bound: 495.4146794
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4144544, upper bound: 495.4147169
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4144180, upper bound: 495.4145134
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4144200, upper bound: 495.4145562
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4143432, upper bound: 495.4145120
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4143414, upper bound: 495.4144682
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4136404, upper bound: 495.4136051
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -495.4135069, upper bound: 495.4135069

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -83.3935242, 89.0365829, -87.0113068, 92.7283020, -176.1218262, 176.0478821
1: -616.7351685, 207.3298187, -641.5713501, 216.3239746, -833.0591431, 848.9009399
2: -334.1997070, 192.2471771, -348.5716248, 200.2107391, -534.4103394, 540.8187866
3: -427.1629944, 154.2363739, -445.1394043, 160.5855865, -587.7485962, 599.3757935
4: -238.4863281, 166.9761505, -248.9928894, 173.7742462, -412.2605591, 415.9689941

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157845, upper bound: 495.4157614
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4158165, upper bound: 495.4157620
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -92.0767212, 97.9184723, -88.7013626, 94.4072647, -186.4839783, 186.6198425
1: -678.4152832, 229.2815552, -653.5552368, 220.6533356, -899.0686035, 882.8367310
2: -369.2738647, 211.7258606, -355.5662842, 203.9700012, -573.2438965, 567.2920532
3: -471.1077881, 169.8008118, -453.8502808, 163.5830536, -634.6906738, 623.6511230
4: -263.9118042, 183.4230804, -254.0701752, 176.8493347, -440.7611389, 437.4931641

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157293, upper bound: 495.4156809
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156420, upper bound: 495.4156130
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -83.3935242, 89.0365829, -100.0263519, 105.8441086, -189.2376404, 189.0629272
1: -616.7351685, 207.3298187, -719.8140259, 248.4512482, -865.1864014, 927.1437378
2: -334.1997070, 192.2471771, -397.5901794, 228.3914642, -562.5910645, 589.8373413
3: -427.1629944, 154.2363739, -502.2337952, 183.2421417, -610.4050903, 656.4700317
4: -238.4863281, 166.9761505, -285.7878113, 198.9984283, -437.4847412, 452.7639771

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4137563, upper bound: 495.4136716
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143448
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -92.0767212, 97.9184723, -101.7525406, 107.5605545, -199.6372681, 199.6710205
1: -678.4152832, 229.2815552, -732.0085449, 252.8869781, -931.3021240, 961.2901001
2: -369.2738647, 211.7258606, -404.7249451, 232.2205505, -601.4943848, 616.4506226
3: -471.1077881, 169.8008118, -511.1030579, 186.2952576, -657.4030762, 680.9038696
4: -263.9118042, 183.4230804, -290.9759827, 202.1451263, -466.0569458, 474.3990479

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4136540, upper bound: 495.4136703
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -84.4675293, 90.1714706, -86.9716721, 92.7193527, -177.1868896, 177.1431427
1: -624.7869873, 210.1310425, -641.8574829, 216.2873230, -841.0742798, 851.9884644
2: -338.9572754, 194.6207581, -348.6205444, 200.1254883, -539.0827637, 543.2413330
3: -432.9501038, 156.1280365, -445.2800598, 160.5431519, -593.4932861, 601.4080200
4: -241.9484100, 168.9341583, -249.0641022, 173.6454773, -415.5938721, 417.9982605

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157407, upper bound: 495.4156238
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157087, upper bound: 495.4155764
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -93.0893936, 98.9997025, -88.6690598, 94.4058609, -187.4951935, 187.6687622
1: -685.9017334, 231.9212036, -653.8798218, 220.6346741, -906.5363770, 885.8009644
2: -373.7428589, 213.9605103, -355.6517639, 203.8992615, -577.6420898, 569.6121216
3: -476.4995422, 171.5928955, -454.0269470, 163.5520935, -640.0514526, 625.6198730
4: -267.1776123, 185.3000641, -254.1669312, 176.7360840, -443.9136658, 439.4669800

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156149, upper bound: 495.4156250
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4155892, upper bound: 495.4155797
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -84.4675293, 90.1714706, -100.3926010, 106.1671371, -190.6346588, 190.5640717
1: -624.7869873, 210.1310425, -721.7380371, 249.3616638, -874.1486816, 931.8688965
2: -338.9572754, 194.6207581, -399.0019836, 229.0488586, -568.0061035, 593.6226196
3: -432.9501038, 156.1280365, -503.6692810, 183.8133850, -616.7634888, 659.7972412
4: -241.9484100, 168.9341583, -286.9118958, 199.5249786, -441.4733887, 455.8460388

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -93.0893936, 98.9997025, -102.1341095, 107.9001083, -200.9894867, 201.1337891
1: -685.9017334, 231.9212036, -734.0285645, 253.8423462, -939.7440796, 965.9497070
2: -373.7428589, 213.9605103, -406.2112732, 232.9138641, -606.6567383, 620.1717529
3: -476.4995422, 171.5928955, -512.6171875, 186.8987122, -663.3980103, 684.2100220
4: -267.1776123, 185.3000641, -292.1506042, 202.7055511, -469.8831482, 477.4506836

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -83.1495667, 88.7888489, -85.6322784, 91.3088074, -174.4583435, 174.4210968
1: -614.8137817, 206.6823120, -632.4382324, 213.0066071, -827.8203125, 839.1205444
2: -333.1367188, 191.6815796, -343.2884216, 197.2170410, -530.3537598, 534.9699707
3: -425.7613220, 153.7995758, -438.4712524, 158.2297058, -583.9910278, 592.2707520
4: -237.7198944, 166.5344086, -245.0929871, 171.1646118, -408.8844910, 411.6273499

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4150059, upper bound: 495.4149601
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149457, upper bound: 495.4148433
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -90.7095261, 96.6783142, -87.3257523, 92.9926224, -183.7021484, 184.0040588
1: -669.9944458, 225.8919983, -644.3334351, 217.3393860, -887.3338013, 870.2254639
2: -363.8521423, 208.9178467, -350.2359009, 200.9761963, -564.8283081, 559.1537476
3: -464.7720032, 167.6038666, -447.1259460, 161.2222900, -625.9942627, 614.7297974
4: -259.8856201, 181.1246643, -250.1490479, 174.2845306, -434.1701050, 431.2737122

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147569, upper bound: 495.4149690
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147074, upper bound: 495.4148519
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -83.1495667, 88.7888489, -98.7767639, 104.5144882, -187.6640625, 187.5655823
1: -614.8137817, 206.6823120, -710.5769043, 245.3585663, -860.1722412, 917.2592163
2: -333.1367188, 191.6815796, -392.4805603, 225.5844269, -558.7211304, 584.1621094
3: -425.7613220, 153.7995758, -495.5682373, 181.0302124, -606.7915039, 649.3677368
4: -237.7198944, 166.5344086, -282.0799255, 196.6259460, -434.3458252, 448.6143188

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140542, upper bound: 495.4140059
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4139210, upper bound: 495.4138441
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -90.7095261, 96.6783142, -100.4956970, 106.2266464, -196.9361725, 197.1740112
1: -669.9944458, 225.8919983, -722.6450195, 249.7634125, -919.7578735, 948.5369873
2: -363.8521423, 208.9178467, -399.5314331, 229.3888702, -593.2410278, 608.4492798
3: -464.7720032, 167.6038666, -504.3525391, 184.0616913, -648.8336792, 671.9563599
4: -259.8856201, 181.1246643, -287.2258301, 199.7845764, -459.6701660, 468.3504944

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4139172, upper bound: 495.4139919
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138212, upper bound: 495.4138408
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -84.1277847, 89.8487244, -85.7188721, 91.4403687, -175.5681458, 175.5675659
1: -622.3063354, 209.2440796, -633.7002563, 213.2749329, -835.5811768, 842.9443359
2: -337.4898682, 193.8820648, -343.8688049, 197.4199829, -534.9098511, 537.7508545
3: -431.0951233, 155.5550995, -439.2847595, 158.4212036, -589.5162964, 594.8398438
4: -240.8812866, 168.3668823, -245.5362091, 171.2866364, -412.1679077, 413.9030762

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4158469, upper bound: 495.4156943
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157968, upper bound: 495.4156566
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -91.6852570, 97.7280579, -87.4222717, 93.1362839, -184.8215332, 185.1502838
1: -677.3110962, 228.4371796, -645.6618652, 217.6327209, -894.9438477, 874.0990601
2: -368.1484375, 211.0962067, -350.8583069, 201.2014160, -569.3496704, 561.9545288
3: -469.9959106, 169.3485413, -447.9974976, 161.4303284, -631.4262695, 617.3460693
4: -263.0168457, 182.9422913, -250.6232758, 174.4284668, -437.4452820, 433.5655212

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157053, upper bound: 495.4156944
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156566, upper bound: 495.4156566
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -84.1277847, 89.8487244, -99.1651001, 104.8706970, -188.9984436, 189.0137939
1: -622.3063354, 209.2440796, -712.6945190, 246.3086243, -868.6149292, 921.9385986
2: -337.4898682, 193.8820648, -393.9373169, 226.3049011, -563.7947388, 587.8193970
3: -431.0951233, 155.5550995, -497.1172791, 181.6457825, -612.7407837, 652.6723633
4: -240.8812866, 168.3668823, -283.2391968, 197.2237396, -438.1050415, 451.6060791

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138538, upper bound: 495.4138288
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138165, upper bound: 495.4137423
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -91.6852570, 97.7280579, -100.9059677, 106.6041946, -198.2894592, 198.6340332
1: -677.3110962, 228.4371796, -724.8949585, 250.7680969, -928.0792236, 953.3320923
2: -368.1484375, 211.0962067, -401.0853882, 230.1584625, -598.3067627, 612.1815796
3: -469.9959106, 169.3485413, -505.9988403, 184.7192535, -654.7151489, 675.3474121
4: -263.0168457, 182.9422913, -288.4531250, 200.4262543, -463.4430542, 471.3953247

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138056, upper bound: 495.4138214
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144900, upper bound: 495.4145349
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144580, upper bound: 495.4145349
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -94.0624466, 99.5080719, -84.7442703, 90.4769287, -184.5393677, 184.2523193
1: -677.2850952, 233.9574432, -624.6894531, 210.4628601, -887.7478638, 858.6469116
2: -373.8485718, 215.0075226, -338.8338318, 195.1003113, -568.9487915, 553.8413696
3: -472.3740845, 172.4977570, -432.8354492, 156.4919281, -628.8659668, 605.3331909
4: -268.8349915, 187.3417053, -242.0598297, 169.6249237, -438.4598999, 429.4015503

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4142512
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4142512
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -97.4328537, 103.1015701, -85.2630768, 91.0219574, -188.4548035, 188.3646088
1: -701.7175903, 242.0646210, -628.5590820, 211.7261658, -913.4437256, 870.6237183
2: -387.3626404, 222.5991669, -340.9848022, 196.2734070, -583.6360474, 563.5838623
3: -489.4045715, 178.5944366, -435.5700684, 157.4351959, -646.8397827, 614.1644897
4: -278.2922058, 193.9222565, -243.5591278, 170.6348114, -448.9270020, 437.4813843

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142033, upper bound: 495.4143539
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142033, upper bound: 495.4143539
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -95.8106079, 101.2488327, -93.3846130, 99.3888779, -195.1994629, 194.6333923
1: -689.5573120, 238.4309387, -686.6988525, 232.3733521, -921.9306641, 925.1297607
2: -381.0275879, 218.8762665, -373.9989929, 214.6445160, -595.6721191, 592.8752441
3: -481.3124390, 175.5816040, -476.9346313, 172.1197815, -653.4322510, 652.5161743
4: -274.0628357, 190.5545654, -267.4910583, 186.1327667, -460.1955261, 458.0456238

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4141785
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4141784
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -99.1810608, 104.8472672, -93.9873123, 100.0126953, -199.1937561, 198.8345795
1: -714.0477295, 246.5407562, -691.0808105, 233.8467255, -947.8944092, 937.6215210
2: -394.5394897, 226.4749908, -376.4795837, 215.9961243, -610.5356445, 602.9545898
3: -498.3539429, 181.6819916, -480.0563965, 173.2042236, -671.5579834, 661.7384033
4: -283.5240784, 197.1485443, -269.2374268, 187.2994080, -470.8234863, 466.3859558

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4142661
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4142661
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -95.1743393, 100.5945511, -84.1974182, 89.8845825, -185.0589294, 184.7919464
1: -685.2960205, 236.8630371, -620.6977539, 209.1443024, -894.4403076, 857.5607300
2: -378.7634888, 217.3093872, -336.7349548, 193.7627716, -572.5260620, 554.0443115
3: -478.0706177, 174.4393005, -430.0692444, 155.4590912, -633.5297241, 604.5085449
4: -272.4121399, 189.3025360, -240.6323090, 168.4118347, -440.8239746, 429.9348145

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141501
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141501
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -97.7576447, 103.3099442, -84.4560852, 90.1572113, -187.9148407, 187.7660217
1: -703.0972290, 242.9924927, -622.9447021, 209.7606659, -912.8579102, 865.9371948
2: -388.9707031, 222.9980011, -337.9386597, 194.3553619, -583.3259888, 560.9365845
3: -490.6489563, 178.9431915, -431.6574402, 155.9313660, -646.5800781, 610.6004639
4: -279.5766907, 194.0973816, -241.4024353, 168.8890991, -448.4657593, 435.4997559

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141490
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141490
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -96.9626389, 102.3774948, -92.8498077, 98.8112335, -195.7738647, 195.2272797
1: -697.8480225, 241.4411774, -682.9046021, 231.0907288, -928.9387207, 924.3457642
2: -386.1035156, 221.2737579, -371.9955139, 213.3415375, -599.4450684, 593.2692871
3: -487.1909790, 177.6015167, -474.2970276, 171.1210327, -658.3118896, 651.8985596
4: -277.7534790, 192.5763397, -266.0812683, 184.9399109, -462.6933594, 458.6575928

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140782, upper bound: 495.4140522
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140782, upper bound: 495.4140522
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -99.5412292, 105.0954285, -93.1692886, 99.1398697, -198.6810913, 198.2646637
1: -715.7137451, 247.5595551, -685.4960327, 231.8585968, -947.5722046, 933.0556030
2: -396.3009949, 226.9573517, -373.4325562, 214.0606232, -610.3616333, 600.3898315
3: -499.7958374, 182.1066132, -476.1528931, 171.6913452, -671.4871826, 658.2595215
4: -284.9136047, 197.3933105, -267.0410767, 185.5337982, -470.4473877, 464.4343872

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140777, upper bound: 495.4140472
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140777, upper bound: 495.4140472
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -93.7450638, 99.1802979, -83.3897095, 89.0731964, -182.8182678, 182.5700073
1: -674.6909180, 233.1021423, -615.7382812, 207.2041931, -881.8951416, 848.8403931
2: -372.4368896, 214.2669220, -333.7265015, 192.1373291, -564.5742188, 547.9934082
3: -470.5045166, 171.9197540, -426.3391113, 154.1683044, -624.6728516, 598.2588501
4: -267.8208008, 186.7701263, -238.2755280, 167.0081635, -434.8289795, 425.0456543

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144519, upper bound: 495.4146794
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144519, upper bound: 495.4146794
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -96.9954987, 102.6449356, -83.7722168, 89.4785767, -186.4740753, 186.4171448
1: -698.2798462, 240.9215088, -618.6599121, 208.1326294, -906.4124756, 859.5814209
2: -385.5005188, 221.5820465, -335.3291626, 193.0048828, -578.5053711, 556.9111938
3: -486.9628601, 177.7921600, -428.4018250, 154.8679504, -641.8306274, 606.1939697
4: -276.9514160, 193.0937500, -239.3758698, 167.7568359, -444.7082520, 432.4696045

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144544, upper bound: 495.4147169
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4147169
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -95.4555893, 100.8789825, -90.9622269, 96.9787979, -192.4343872, 191.8412170
1: -686.6740723, 237.4833527, -671.0132446, 226.4459076, -913.1199951, 908.4965820
2: -379.4509888, 218.0464783, -364.4942932, 209.4122009, -588.8630981, 582.5407715
3: -479.2211914, 174.9310760, -465.4107971, 168.0008545, -647.2219849, 640.3418579
4: -272.9408569, 189.8968506, -260.4682922, 181.6331787, -454.5739746, 450.3651428

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4145134
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144180, upper bound: 495.4145134
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -98.7163239, 104.3602524, -91.3357086, 97.3774796, -196.0937958, 195.6959534
1: -710.3906860, 245.3320160, -673.9136963, 227.3468781, -937.7375488, 919.2457275
2: -392.5589905, 225.3939056, -366.0620728, 210.2654572, -602.8244629, 591.4557495
3: -495.7640381, 180.8292999, -467.4552307, 168.6866760, -664.4506836, 648.2844849
4: -282.1028137, 196.2624817, -261.5397034, 182.3691254, -464.4719238, 457.8020935

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144200, upper bound: 495.4145562
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144200, upper bound: 495.4145562
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -94.8167801, 100.2254562, -82.9283295, 88.5842590, -183.4010315, 183.1537781
1: -682.4131470, 235.9046783, -612.4281006, 206.0847168, -888.4978638, 848.3327637
2: -377.1776123, 216.4719086, -331.9933167, 191.0076294, -568.1851196, 548.4652100
3: -475.9878845, 173.7890320, -424.0395508, 153.2991638, -629.2870483, 597.8285522
4: -271.2825623, 188.6606445, -237.0956573, 165.9823761, -437.2649231, 425.7562561

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4145120
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143432, upper bound: 495.4145120
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -97.1539764, 102.7001801, -83.0699158, 88.7369003, -185.8908691, 185.7700806
1: -698.4937134, 241.4297943, -613.8479614, 206.4155121, -904.9091187, 855.2777100
2: -386.3923645, 221.6345520, -332.7180176, 191.3371124, -577.7294922, 554.3525391
3: -487.3369446, 177.8703766, -425.0419922, 153.5640869, -640.9010010, 602.9123535
4: -277.7228088, 193.0055847, -237.5198822, 166.2381287, -443.9608765, 430.5254211

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4144682
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143414, upper bound: 495.4144682
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -98.2204819, 103.6226501, -89.4449310, 95.3808823, -193.6013641, 193.0675812
1: -704.7350464, 244.2698517, -660.2431030, 222.7252960, -927.4602051, 904.5129395
2: -390.7516785, 223.8087006, -358.7164001, 205.8770142, -596.6286621, 582.5250854
3: -492.2799377, 179.6109619, -457.9817810, 165.2118378, -657.4916992, 637.5927734
4: -281.0237427, 194.7337189, -256.3164673, 178.5086517, -459.5323181, 451.0501709

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4136404, upper bound: 495.4136051
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4136404, upper bound: 495.4136051
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -100.4968796, 106.2339554, -89.1064529, 94.9624634, -195.4593506, 195.3404083
1: -722.3804321, 249.7090607, -657.1383667, 221.8153229, -944.1956787, 906.8474121
2: -399.7193909, 229.3199310, -357.2251282, 205.0100098, -604.7293701, 586.5450439
3: -503.9279175, 184.0366058, -455.9029236, 164.4986115, -668.4265137, 639.9395142
4: -287.0542297, 199.6714630, -255.2903137, 177.7408905, -464.7951050, 454.9617920

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4135069, upper bound: 495.4135069
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4135069, upper bound: 495.4135069
time: 0.98 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.47 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4157845, upper bound: 495.4157614
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4158165, upper bound: 495.4157620
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4157293, upper bound: 495.4156809
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4156420, upper bound: 495.4156130
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143448
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143034, upper bound: 495.4143447
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4157407, upper bound: 495.4156238
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4157087, upper bound: 495.4155764
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4156149, upper bound: 495.4156250
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4155892, upper bound: 495.4155797
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143421, upper bound: 495.4143292
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4150059, upper bound: 495.4149601
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4149457, upper bound: 495.4148433
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4147569, upper bound: 495.4149690
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4147074, upper bound: 495.4148519
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140542, upper bound: 495.4140059
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4139210, upper bound: 495.4138441
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4139172, upper bound: 495.4139919
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4138212, upper bound: 495.4138408
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4158469, upper bound: 495.4156943
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4157968, upper bound: 495.4156566
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4157053, upper bound: 495.4156944
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4156566, upper bound: 495.4156566
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4138538, upper bound: 495.4138288
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4138165, upper bound: 495.4137423
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144900, upper bound: 495.4145349
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144580, upper bound: 495.4145349
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4142512
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4142512
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4142033, upper bound: 495.4143539
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4142033, upper bound: 495.4143539
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4141785
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4141784
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4142661
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4142661
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141501
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141501
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141490
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4141490
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140782, upper bound: 495.4140522
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140782, upper bound: 495.4140522
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140777, upper bound: 495.4140472
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140777, upper bound: 495.4140472
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144519, upper bound: 495.4146794
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144519, upper bound: 495.4146794
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144544, upper bound: 495.4147169
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141602, upper bound: 495.4147169
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4141594, upper bound: 495.4145134
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144180, upper bound: 495.4145134
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144200, upper bound: 495.4145562
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4144200, upper bound: 495.4145562
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4145120
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143432, upper bound: 495.4145120
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4140861, upper bound: 495.4144682
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4143414, upper bound: 495.4144682
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4136404, upper bound: 495.4136051
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4136404, upper bound: 495.4136051
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4135069, upper bound: 495.4135069
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -495.4135069, upper bound: 495.4135069

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -82.1456451, 87.6649323, -84.4311905, 89.9280853, -172.0737305, 172.0960999
1: -607.3638306, 204.2498322, -622.6676025, 209.9906006, -817.3544312, 826.9173584
2: -329.1680908, 189.3492279, -338.3735962, 194.2957458, -523.4638062, 527.7227783
3: -420.6973877, 151.9087219, -432.0761108, 155.8410034, -576.5383301, 583.9848633
4: -234.9110107, 164.4322052, -241.7219391, 168.5518799, -403.4628906, 406.1541138

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157404, upper bound: 495.4155861
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157399, upper bound: 495.4155835
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -82.1988449, 87.6915283, -90.1541672, 96.0639725, -178.2628174, 177.8457031
1: -606.3214111, 204.3459930, -663.1215210, 224.2644196, -830.5858154, 867.4674683
2: -329.1330872, 189.4078217, -361.0358887, 207.5007324, -536.6337891, 550.4436646
3: -420.0945740, 151.9515381, -460.3447571, 166.3946533, -586.4891968, 612.2962036
4: -234.9892120, 164.5832367, -258.0718689, 180.1872864, -415.1764832, 422.6550903

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157505, upper bound: 495.4155860
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157505, upper bound: 495.4155814
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -91.4662933, 97.2492142, -87.2814560, 92.8340683, -184.3003540, 184.5306702
1: -673.7709961, 227.7635040, -641.7135620, 217.1164398, -890.8874512, 869.4769897
2: -366.8383484, 210.2893219, -349.6722107, 200.5722656, -567.4105835, 559.9615479
3: -467.9310303, 168.6487122, -445.8273315, 160.8435516, -628.7745972, 614.4760742
4: -262.1795044, 182.1620178, -249.9685669, 173.9154510, -436.0948486, 432.1305847

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157076, upper bound: 495.4156156
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157157, upper bound: 495.4156754
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -91.8166199, 97.6309586, -88.3258514, 93.9934998, -185.8100739, 185.9567871
1: -676.4699707, 228.6351624, -650.7705078, 219.7205658, -896.1905518, 879.4056396
2: -368.2498779, 211.1069641, -354.0934448, 203.0817108, -571.3315430, 565.2004395
3: -469.7727051, 169.3014069, -451.9333191, 162.8669128, -632.6396484, 621.2347412
4: -263.1791687, 182.8706055, -253.0117950, 176.0560760, -439.2351685, 435.8823242

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156243, upper bound: 495.4155254
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156299, upper bound: 495.4155896
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -83.3935242, 89.0365829, -97.7803268, 103.5871124, -186.9806366, 186.8169098
1: -616.7351685, 207.3298187, -703.7860718, 242.6765289, -859.4116821, 911.1157837
2: -334.1997070, 192.2471771, -388.2923889, 223.3814850, -557.5810547, 580.5395508
3: -427.1629944, 154.2363739, -490.5957642, 179.2505035, -606.4133301, 644.8320923
4: -238.4863281, 166.9761505, -279.0299072, 194.8641968, -433.3505249, 446.0060425

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143913, upper bound: 495.4143460
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143913, upper bound: 495.4143461
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -83.3935242, 89.0365829, -106.4699478, 112.5509186, -195.9444427, 195.5065308
1: -616.7351685, 207.3298187, -766.2641602, 264.8076172, -881.5427856, 973.5938721
2: -334.1997070, 192.2471771, -423.6825256, 243.0426941, -577.2423706, 615.9296875
3: -427.1629944, 154.2363739, -535.0036621, 194.9680481, -622.1310425, 689.2400513
4: -238.4863281, 166.9761505, -304.6231079, 211.4498749, -449.9361877, 471.5992126

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143913, upper bound: 495.4143461
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143913, upper bound: 495.4143460
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -92.0767212, 97.9184723, -97.7803268, 103.5871124, -195.6638336, 195.6987915
1: -678.4152832, 229.2815552, -703.7860718, 242.6765289, -921.0917969, 933.0676270
2: -369.2738647, 211.7258606, -388.2923889, 223.3814850, -592.6553345, 600.0179443
3: -471.1077881, 169.8008118, -490.5957642, 179.2505035, -650.3582153, 660.3966064
4: -263.9118042, 183.4230804, -279.0299072, 194.8641968, -458.7760010, 462.4530029

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142820, upper bound: 495.4143386
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142820, upper bound: 495.4143386
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -92.0767212, 97.9184723, -106.4699478, 112.5509186, -204.6276245, 204.3884277
1: -678.4152832, 229.2815552, -766.2641602, 264.8076172, -943.2229004, 995.5457153
2: -369.2738647, 211.7258606, -423.6825256, 243.0426941, -612.3165283, 635.4080811
3: -471.1077881, 169.8008118, -535.0036621, 194.9680481, -666.0758057, 704.8044434
4: -263.9118042, 183.4230804, -304.6231079, 211.4498749, -475.3616943, 488.0462036

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142820, upper bound: 495.4143386
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142820, upper bound: 495.4143448
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.9273529, 89.5829239, -85.4269333, 91.0160904, -174.9434509, 175.0098114
1: -620.7719727, 208.7960205, -629.3035278, 212.4340820, -833.2059326, 838.0994263
2: -336.8411865, 193.3575592, -342.2850952, 196.4536591, -533.2947998, 535.6426392
3: -430.2026978, 155.1085205, -436.7432861, 157.5812836, -587.7839966, 591.8518066
4: -240.4320374, 167.8116760, -244.6146240, 170.4416809, -410.8737183, 412.4263000

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157067, upper bound: 495.4155704
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157067, upper bound: 495.4156005
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -84.1174011, 89.7784958, -86.4030914, 92.0874786, -176.2048645, 176.1815796
1: -622.0245361, 209.2519836, -637.4918823, 214.8634186, -836.8878784, 846.7437744
2: -337.5271912, 193.7772980, -346.3357239, 198.7667694, -536.2939453, 540.1129761
3: -431.0579529, 155.4505005, -442.2822266, 159.4535828, -590.5115356, 597.7327271
4: -240.9403076, 168.1950378, -247.4381866, 172.4479675, -413.3882751, 415.6332397

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156943, upper bound: 495.4154921
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156943, upper bound: 495.4155373
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -92.5699234, 98.4329453, -87.1236267, 92.7032623, -185.2731934, 185.5565338
1: -682.0200195, 230.6385956, -641.3147583, 216.7794342, -898.7993774, 871.9533691
2: -371.7113342, 212.7438202, -349.3033447, 200.2305145, -571.9418335, 562.0471802
3: -473.8509827, 170.6117401, -445.4825134, 160.5907288, -634.4415283, 616.0942383
4: -265.7235107, 184.2175293, -249.7108307, 173.5500793, -439.2735291, 433.9283447

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4155996, upper bound: 495.4155860
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4155996, upper bound: 495.4156133
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -92.7244797, 98.5920258, -88.1051331, 93.7785568, -186.5030212, 186.6971130
1: -683.0306396, 231.0039520, -649.5386353, 219.2212830, -902.2519531, 880.5424194
2: -372.2545471, 213.0830841, -353.3854370, 202.5518494, -574.8063354, 566.4685059
3: -474.5351868, 170.8884125, -451.0487061, 162.4698029, -637.0048828, 621.9371338
4: -266.1288452, 184.5325623, -252.5556946, 175.5480804, -441.6768799, 437.0882568

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4155782, upper bound: 495.4155071
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4155782, upper bound: 495.4155510
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -84.4675293, 90.1714706, -98.1343155, 103.9021988, -188.3697205, 188.3057861
1: -624.7869873, 210.1310425, -705.6849976, 243.5524902, -868.3394775, 915.8158569
2: -338.9572754, 194.6207581, -389.6584778, 224.0160522, -562.9733276, 584.2792358
3: -432.9501038, 156.1280365, -492.0138245, 179.8005524, -612.7506714, 648.1418457
4: -241.9484100, 168.9341583, -280.1228027, 195.3724213, -437.3208313, 449.0569153

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144550, upper bound: 495.4143316
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144550, upper bound: 495.4143416
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -84.4675293, 90.1714706, -106.7867508, 112.8346252, -197.3021545, 196.9582214
1: -624.7869873, 210.1310425, -768.1140747, 265.6062012, -890.3931885, 978.2449341
2: -338.9572754, 194.6207581, -424.9284668, 243.6081848, -582.5654297, 619.5491943
3: -432.9501038, 156.1280365, -536.3448486, 195.4741364, -628.4242554, 692.4728394
4: -241.9484100, 168.9341583, -305.5875854, 211.8855438, -453.8339539, 474.5217285

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144550, upper bound: 495.4143316
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4144550, upper bound: 495.4143416
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.0893936, 98.9997025, -98.1343155, 103.9021988, -196.9915771, 197.1340179
1: -685.9017334, 231.9212036, -705.6849976, 243.5524902, -929.4541626, 937.6061401
2: -373.7428589, 213.9605103, -389.6584778, 224.0160522, -597.7589111, 603.6188965
3: -476.4995422, 171.5928955, -492.0138245, 179.8005524, -656.3000488, 663.6066895
4: -267.1776123, 185.3000641, -280.1228027, 195.3724213, -462.5499878, 465.4228516

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143309, upper bound: 495.4143236
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143309, upper bound: 495.4143292
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.0893936, 98.9997025, -106.7867508, 112.8346252, -205.9240112, 205.7864532
1: -685.9017334, 231.9212036, -768.1140747, 265.6062012, -951.5079346, 1000.0352173
2: -373.7428589, 213.9605103, -424.9284668, 243.6081848, -617.3510742, 638.8888550
3: -476.4995422, 171.5928955, -536.3448486, 195.4741364, -671.9735107, 707.9377441
4: -267.1776123, 185.3000641, -305.5875854, 211.8855438, -479.0631714, 490.8876343

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4142820, upper bound: 495.4143236
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4143309, upper bound: 495.4143237
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -81.9906158, 87.5796127, -83.9734573, 89.5878448, -171.5784454, 171.5530701
1: -605.8366699, 203.7156677, -619.4334717, 208.7919006, -814.6285400, 823.1491699
2: -328.4120178, 189.0596161, -336.4824829, 193.4797821, -521.8917847, 525.5420532
3: -419.6186829, 151.6552582, -429.8921814, 155.1859589, -574.8045654, 581.5474243
4: -234.4029846, 164.3563690, -240.4285278, 168.0707550, -402.4737549, 404.7849121

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149764, upper bound: 495.4148763
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149859, upper bound: 495.4149502
time: 3.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -80.6512375, 86.2526093, -84.3532867, 90.1924820, -170.8437195, 170.6058960
1: -596.8311768, 200.4649048, -623.7106323, 209.7607574, -806.5919189, 824.1754150
2: -323.0969849, 186.1069794, -338.1374817, 194.5682373, -517.6652222, 524.2444458
3: -413.1379089, 149.3313599, -432.1332092, 156.1189270, -569.2567139, 581.4645386
4: -230.5420837, 161.8113403, -241.3581390, 169.1366119, -399.6787109, 403.1694946

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149190, upper bound: 495.4147495
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4149328, upper bound: 495.4148335
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -89.5396423, 95.4595261, -85.6555405, 91.2670822, -180.8067322, 181.1150513
1: -660.9591064, 222.9024963, -631.3317871, 213.1005096, -874.0596313, 854.2342529
2: -359.0946350, 206.2737122, -343.4085693, 197.2193756, -556.3140259, 549.6822510
3: -458.5811157, 165.4415131, -438.5712280, 158.1629181, -616.7438965, 604.0127563
4: -256.5436707, 178.9235229, -245.4670868, 171.1689301, -427.7125854, 424.3906250

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147360, upper bound: 495.4148937
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147360, upper bound: 495.4149661
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -88.2111206, 94.1430511, -86.0439377, 91.8808823, -180.0919647, 180.1869812
1: -652.0000000, 219.6769257, -635.5845337, 214.0894623, -866.0894775, 855.2614746
2: -353.8005066, 203.3474731, -345.0904236, 198.3283844, -552.1289062, 548.4378052
3: -452.1293945, 163.1340332, -440.7983704, 159.1097565, -611.2391357, 603.9323730
4: -252.6995850, 176.4060364, -246.4139709, 172.2600403, -424.9596252, 422.8199768

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147773, upper bound: 495.4147581
time: 2.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4147030, upper bound: 495.4148456
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -80.5124741, 86.0098114, -95.7619019, 101.2649536, -181.7774353, 181.7717133
1: -596.1480103, 200.1852264, -688.1715698, 237.8845062, -834.0324707, 888.3568115
2: -322.8490601, 185.6872101, -380.4059753, 218.5931396, -541.4421997, 566.0932007
3: -412.7777100, 148.9892578, -480.0433655, 175.4355927, -588.2133179, 629.0324707
4: -230.2955170, 161.2815704, -273.4135742, 190.5127411, -420.8082275, 434.6951294

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4136463, upper bound: 495.4136969
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4136463, upper bound: 495.4140059
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.3537750, 86.8417740, -100.9383545, 106.9987640, -188.3525391, 187.7801208
1: -601.2239990, 202.2021484, -727.6601562, 250.5857849, -851.8098145, 929.8623047
2: -325.9946594, 187.4929810, -401.1294861, 230.8689575, -556.8636475, 588.6224365
3: -416.4656067, 150.4191284, -506.7908630, 185.2861938, -601.7518311, 657.2099609
4: -232.6330109, 162.8636169, -287.8343506, 201.3775330, -434.0105286, 450.6978760

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4137301, upper bound: 495.4136881
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138204, upper bound: 495.4137580
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -88.0383148, 93.8624954, -97.4808350, 102.9720764, -191.0103302, 191.3432922
1: -650.9976196, 219.2974091, -700.1837769, 242.2887421, -893.2863159, 919.4811401
2: -353.3913879, 202.8391724, -387.4418640, 222.3940430, -575.7853394, 590.2810059
3: -451.5660706, 162.7263336, -488.7943420, 178.4638977, -630.0299683, 651.5205688
4: -252.3448181, 175.8048553, -278.5543823, 193.6690063, -446.0137634, 454.3592529

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4135498, upper bound: 495.4136953
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4135498, upper bound: 495.4139919
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -88.8819275, 94.6991425, -102.6050339, 108.6492691, -197.5311584, 197.3041687
1: -656.1378784, 221.3237915, -739.2744141, 254.8557739, -910.9934692, 960.5981445
2: -356.5531921, 204.6563568, -407.9488831, 234.5537872, -591.1069946, 612.6052246
3: -455.2868042, 164.1625824, -515.2566528, 188.2242889, -643.5111084, 679.4192505
4: -254.6905975, 177.3996124, -292.8153992, 204.4020996, -459.0926819, 470.2150269

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4135346, upper bound: 495.4134434
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4132227, upper bound: 495.4133067
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.5850983, 89.2572098, -84.0671921, 89.6346664, -173.2197571, 173.3243866
1: -618.2677002, 207.9020233, -620.3942261, 209.1561737, -827.4238281, 828.2962646
2: -335.3610840, 192.6128082, -337.1176758, 193.5282288, -528.8892822, 529.7304688
3: -428.3307800, 154.5300903, -430.2167969, 155.2847443, -583.6155396, 584.7468872
4: -239.3568268, 167.2397156, -240.7874298, 167.9152985, -407.2721252, 408.0270691

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157956, upper bound: 495.4156402
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4158133, upper bound: 495.4156809
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -83.7818451, 89.4599686, -85.1525421, 90.8093033, -174.5911407, 174.6125183
1: -619.5738525, 208.3749695, -629.3350220, 211.8562012, -831.4300537, 837.7099609
2: -336.0769348, 193.0474701, -341.5891724, 196.0642395, -532.1411743, 534.6365356
3: -429.2242737, 154.8848877, -436.2910461, 157.3319397, -586.5562134, 591.1758423
4: -239.8854980, 167.6350708, -243.9141846, 170.0900269, -409.9755249, 411.5492554

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157736, upper bound: 495.4155940
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4157791, upper bound: 495.4156411
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -91.1644821, 97.1581345, -85.7705002, 91.3309250, -182.4954071, 182.9286346
1: -673.3949585, 227.1496277, -632.3489380, 213.5120392, -886.9069824, 859.4984741
2: -366.1053162, 209.8722229, -344.1026917, 197.3121033, -563.4173584, 553.9749146
3: -467.3251953, 168.3619080, -438.9229126, 158.2937927, -625.6188354, 607.2847900
4: -261.5568848, 181.8560028, -245.8699951, 171.0630188, -432.6198120, 427.7260132

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156736, upper bound: 495.4156453
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156864, upper bound: 495.4156844
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -91.3263702, 97.3252487, -86.8607788, 92.5098801, -183.8362427, 184.1860352
1: -674.4799805, 227.5353546, -641.3287964, 216.2253571, -890.7053223, 868.8641357
2: -366.6852722, 210.2302246, -348.6008606, 199.8565063, -566.5417480, 558.8309937
3: -468.0601807, 168.6536865, -445.0289917, 160.3485718, -628.4086304, 613.6826172
4: -261.9854126, 182.1829224, -249.0179291, 173.2408600, -435.2262573, 431.2008362

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156375, upper bound: 495.4156034
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4156439, upper bound: 495.4156439
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -82.2058868, 87.8266525, -96.8327942, 102.3466949, -184.5525818, 184.6594391
1: -608.4161987, 204.4944000, -695.2703247, 240.5321045, -848.9482422, 899.7647095
2: -329.9023743, 189.4938812, -384.6160278, 220.8870087, -550.7893677, 574.1099243
3: -421.4460144, 152.0419769, -485.0973816, 177.2973480, -598.7433472, 637.1393433
4: -235.4308319, 164.5513763, -276.5522156, 192.4727936, -427.9036255, 441.1035767

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138030, upper bound: 495.4137664
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138165, upper bound: 495.4137423
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138165, upper bound: 495.4137423
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.0847015, 86.5501938, -100.4970016, 106.5005188, -187.5852203, 187.0471954
1: -599.3234863, 201.6219482, -724.2965698, 249.5005798, -848.8240356, 925.9185181
2: -325.2566223, 186.7895203, -399.3910217, 229.7456055, -555.0021973, 586.1805420
3: -415.2701111, 149.8536530, -504.4582825, 184.4203186, -599.6904297, 654.3119507
4: -232.1641083, 162.1823883, -286.6341553, 200.3397522, -432.5038452, 448.8164978

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138165, upper bound: 495.4137423
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -495.4138165, upper bound: 495.4137423
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -90.7198563, 96.7193069, -96.2812805, 101.6621323, -192.3819580, 193.0005798
1: -670.2472534, 226.0417023, -691.4762573, 239.4884033, -909.7356567, 917.5179443
2: -364.2188110, 208.9071960, -382.6340637, 219.6613312, -583.8801270, 591.5412598
3: -465.0106201, 167.5866852, -482.7075806, 176.2904053, -641.3009644, 650.2942505
4: -260.2291870, 181.0478058, -275.3824768, 191.2684479, -451.4976196, 456.4302673

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.61 + 416.74 = 420.34 seconds
