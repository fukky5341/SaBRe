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
execution time: IAR + RelationalAnalysis = 2.28 + 1.71 = 4.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -56.4772858, upper bound: 56.4772858

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4725893, upper bound: 56.4508342
time: 0.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4725893, upper bound: 56.4755533
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 4, lower bound: -56.4725893, upper bound: 56.4508342
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 4, lower bound: -56.4725893, upper bound: 56.4755533

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -137.1921082, 248.1778717, -150.7717438, 278.2929382, -415.4850464, 398.9496155
1: -27.2023220, 31.5797462, -30.6839504, 35.1584358, -62.3607559, 62.2636833
2: -19.5491886, 32.3669205, -21.8219566, 36.0254250, -55.5746117, 54.1888771
3: -18.5265369, 54.1269684, -20.6292686, 60.2387085, -78.7652435, 74.7562332
4: -15.9609337, 40.2531700, -17.8182507, 44.7172356, -60.6781654, 58.0714149

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4596711, upper bound: 56.4272137
time: 0.57 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4725893, upper bound: 56.4508342
time: 0.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -141.0739594, 258.2863159, -150.7717438, 278.2929382, -419.3668823, 409.0580444
1: -28.4475994, 32.6651268, -30.6839504, 35.1584358, -63.6060219, 63.3490753
2: -20.2786674, 33.4789200, -21.8219566, 36.0254250, -56.3040924, 55.3008766
3: -19.1661892, 56.0456581, -20.6292686, 60.2387085, -79.4048996, 76.6749268
4: -16.5348206, 41.5980759, -17.8182507, 44.7172356, -61.2520523, 59.4163246

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4719766, upper bound: 56.4515972
time: 0.58 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.98 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 4, lower bound: -56.4596711, upper bound: 56.4272137
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 4, lower bound: -56.4725893, upper bound: 56.4508342
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 4, lower bound: -56.4719766, upper bound: 56.4515972
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -129.7380981, 231.8643494, -149.7102356, 276.0789490, -405.8170471, 381.5745850
1: -25.3030415, 29.5850067, -30.4246235, 34.8850365, -60.1880798, 60.0096245
2: -18.2123375, 30.4423599, -21.6499100, 35.7492447, -53.9615822, 52.0922699
3: -17.2143021, 50.8256493, -20.4665165, 59.7822380, -76.9965363, 71.2921448
4: -14.8367062, 38.1045609, -17.6758347, 44.3853760, -59.2220802, 55.7803917

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4421469, upper bound: 56.4189087
time: 0.59 seconds

## Relational analysis of NS_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4459906, upper bound: 56.4263502
time: 0.58 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4459906, upper bound: 56.4272137
time: 0.59 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -130.6304169, 235.2711639, -150.7717438, 278.2929382, -408.9233093, 386.0429077
1: -25.7617416, 29.9647408, -30.6839504, 35.1584358, -60.9201736, 60.6486893
2: -18.5520706, 30.7307091, -21.8219566, 36.0254250, -54.5774918, 52.5526657
3: -17.5833778, 51.4350700, -20.6292686, 60.2387085, -77.8220825, 72.0643387
4: -15.1328659, 38.2542572, -17.8182507, 44.7172356, -59.8501015, 56.0725021

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4622555, upper bound: 56.4352935
time: 0.58 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4624494, upper bound: 56.4241431
time: 0.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -141.0739594, 258.2863159, -149.7285309, 276.2070312, -417.2809753, 408.0148010
1: -28.4475994, 32.6651268, -30.4487743, 34.8926010, -63.3402023, 63.1138992
2: -20.2786674, 33.4789200, -21.6595669, 35.7649422, -56.0436096, 55.1384850
3: -19.1661892, 56.0456581, -20.4798374, 59.8106728, -78.9768600, 76.5254974
4: -16.5348206, 41.5980759, -17.6845875, 44.3923340, -60.9271545, 59.2826538

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664838, upper bound: 56.4509004
time: 0.62 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4718914, upper bound: 56.4515657
time: 0.79 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -141.0622406, 258.2618103, -156.3780212, 288.2749023, -429.3371277, 414.6398315
1: -28.4448414, 32.6621208, -31.7632923, 36.4643555, -64.9091873, 64.4254150
2: -20.2767963, 33.4758530, -22.6010170, 37.3600349, -57.6368256, 56.0768700
3: -19.1644363, 56.0405769, -21.3841972, 62.4067574, -81.5711975, 77.4247742
4: -16.5332699, 41.5942993, -18.4631386, 46.3920479, -62.9253044, 60.0574341

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4411746, upper bound: 56.4480086
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.74 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4459906, upper bound: 56.4263502
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4459906, upper bound: 56.4272137
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4622555, upper bound: 56.4352935
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4624494, upper bound: 56.4241431
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4664838, upper bound: 56.4509004
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4718914, upper bound: 56.4515657
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4411746, upper bound: 56.4480086
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -129.7380981, 231.8643494, -136.2755585, 246.2902679, -376.0283203, 368.1398926
1: -25.3030415, 29.5850067, -26.9791889, 31.3460903, -56.6491318, 56.5641937
2: -18.2123375, 30.4423599, -19.4025211, 32.1306343, -50.3429718, 49.8448639
3: -17.2143021, 50.8256493, -18.3838043, 53.7354774, -70.9497833, 69.2094574
4: -14.8367062, 38.1045609, -15.8388062, 39.9675217, -54.8042221, 53.9433670

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4429269, upper bound: 56.4155910
time: 0.51 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4459465, upper bound: 56.4263273
time: 0.59 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -129.7380981, 231.8643494, -139.8852844, 255.8192596, -385.5573730, 371.7495117
1: -25.3030415, 29.5850067, -28.1629448, 32.3608208, -57.6638641, 57.7479401
2: -18.2123375, 30.4423599, -20.0881538, 33.1676636, -51.3800011, 50.5305099
3: -17.2143021, 50.8256493, -18.9853401, 55.5363846, -72.7506866, 69.8109894
4: -14.8367062, 38.1045609, -16.3765030, 41.2219124, -56.0586166, 54.4810638

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4429269, upper bound: 56.4167407
time: 0.59 seconds

## Relational analysis of NS_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4459465, upper bound: 56.4272137
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: -118.0183487, 212.4245605, -150.4909515, 277.7034912, -395.7218018, 362.9154968
1: -23.1348152, 26.9111767, -30.6177006, 35.0837975, -58.2186127, 57.5288773
2: -16.6538963, 27.7483234, -21.7763500, 35.9525337, -52.6064301, 49.5246658
3: -15.6745472, 46.5898361, -20.5878410, 60.1179695, -75.7925110, 67.1776733
4: -13.5144119, 34.6450195, -17.7808437, 44.6271095, -58.1415215, 52.4258575

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_A1_B1

### Relational analysis result of NS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4569483, upper bound: 56.4343766
time: 0.57 seconds

## Relational analysis of NS_A1_A2_A1_B2

### Relational analysis result of NS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4546127, upper bound: 56.4343720
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -123.6277847, 220.7419434, -150.7717438, 278.2929382, -401.9207153, 371.5136719
1: -24.1226845, 28.1704769, -30.6839504, 35.1584358, -59.2811203, 58.8544273
2: -17.4426174, 28.8812294, -21.8219566, 36.0254250, -53.4680405, 50.7031860
3: -16.5396194, 48.4398308, -20.6292686, 60.2387085, -76.7783279, 69.0690994
4: -14.2134304, 36.0034599, -17.8182507, 44.7172356, -58.9306641, 53.8217087

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4567187, upper bound: 56.4231697
time: 0.58 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4531780, upper bound: 56.4223792
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -141.0739594, 258.2863159, -146.7868042, 271.2817078, -412.3556519, 405.0731201
1: -28.4475994, 32.6651268, -29.8981628, 34.2219620, -62.6695557, 62.5632820
2: -20.2786674, 33.4789200, -21.2400036, 35.1723442, -55.4510117, 54.7189255
3: -19.1661892, 56.0456581, -20.0754395, 58.7820320, -77.9482193, 76.1210861
4: -16.5348206, 41.5980759, -17.3301086, 43.6437874, -60.1786041, 58.9281807

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4652255, upper bound: 56.4384253
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664838, upper bound: 56.4509004
time: 0.65 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4664838, upper bound: 56.4509004
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -141.0739594, 258.2863159, -148.8385620, 274.2528381, -415.3267822, 407.1248779
1: -28.4475994, 32.6651268, -30.2171135, 34.6515656, -63.0991516, 62.8822403
2: -20.2786674, 33.4789200, -21.5122261, 35.5275002, -55.8061638, 54.9911461
3: -19.1661892, 56.0456581, -20.3489761, 59.4197006, -78.5858917, 76.3946381
4: -16.5348206, 41.5980759, -17.5647049, 44.1017761, -60.6365929, 59.1627731

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4246259, upper bound: 56.4423686
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4689175, upper bound: 56.4505319
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -141.0622406, 258.2618103, -153.5695343, 283.6153564, -424.6776123, 411.8313293
1: -28.4448414, 32.6621208, -31.2477360, 35.8313904, -64.2762222, 63.9098434
2: -20.2767963, 33.4758530, -22.2131824, 36.8044777, -57.0812683, 55.6890335
3: -19.1644363, 56.0405769, -20.9997807, 61.4355774, -80.6000137, 77.0403595
4: -16.5332699, 41.5942993, -18.1305389, 45.6889343, -62.2221947, 59.7248383

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4406556, upper bound: 56.4354755
time: 0.60 seconds

## Relational analysis of NS_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4411746, upper bound: 56.4480086
time: 0.65 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4411746, upper bound: 56.4480086
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -141.0622406, 258.2618103, -155.4292450, 286.1744995, -427.2367554, 413.6910095
1: -28.4448414, 32.6621208, -31.5192776, 36.2066917, -64.6515350, 64.1813965
2: -20.2767963, 33.4758530, -22.4438629, 37.1029854, -57.3797798, 55.9197159
3: -19.1644363, 56.0405769, -21.2429676, 61.9818573, -81.1462936, 77.2835388
4: -16.5332699, 41.5942993, -18.3347549, 46.0759964, -62.6092567, 59.9290504

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.52 seconds
NS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4429269, upper bound: 56.4155910
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4459465, upper bound: 56.4263273
NS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4429269, upper bound: 56.4167407
NS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4459465, upper bound: 56.4272137
NS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4569483, upper bound: 56.4343766
NS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4546127, upper bound: 56.4343720
NS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4567187, upper bound: 56.4231697
NS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4531780, upper bound: 56.4223792
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4664838, upper bound: 56.4509004
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4664838, upper bound: 56.4509004
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4246259, upper bound: 56.4423686
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4689175, upper bound: 56.4505319
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4411746, upper bound: 56.4480086
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4411746, upper bound: 56.4480086
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 4, lower bound: -56.4491845, upper bound: 56.4491845

## BFS NS instance: NS_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -131.4675751, 231.1583557, -136.2755585, 246.2902679, -377.7578125, 367.4338989
1: -25.0598774, 29.6741943, -26.9791889, 31.3460903, -56.4059639, 56.6533813
2: -18.2806129, 30.4280529, -19.4025211, 32.1306343, -50.4112473, 49.8305626
3: -17.2635632, 50.9579620, -18.3838043, 53.7354774, -70.9990387, 69.3417664
4: -14.8596067, 38.0985489, -15.8388062, 39.9675217, -54.8271217, 53.9373550

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4423234, upper bound: 56.4139676
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_A1_A2

### Relational analysis result of NS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4423234, upper bound: 56.4155910
time: 0.61 seconds

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -126.5899277, 225.5180664, -136.2755585, 246.2902679, -372.8801880, 361.7936401
1: -24.5684433, 28.8065701, -26.9791889, 31.3460903, -55.9145355, 55.7857590
2: -17.7349644, 29.6324329, -19.4025211, 32.1306343, -49.8656006, 49.0349464
3: -16.7349968, 49.4946289, -18.3838043, 53.7354774, -70.4704742, 67.8784180
4: -14.4265404, 37.1043396, -15.8388062, 39.9675217, -54.3940620, 52.9431458

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4441838, upper bound: 56.4245651
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4370419, upper bound: 56.4073689
time: 0.58 seconds

## BFS NS instance: NS_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -131.4675751, 231.1583557, -139.8852844, 255.8192596, -387.2868347, 371.0435486
1: -25.0598774, 29.6741943, -28.1629448, 32.3608208, -57.4206886, 57.8371315
2: -18.2806129, 30.4280529, -20.0881538, 33.1676636, -51.4482727, 50.5162048
3: -17.2635632, 50.9579620, -18.9853401, 55.5363846, -72.7999496, 69.9432983
4: -14.8596067, 38.0985489, -16.3765030, 41.2219124, -56.0815201, 54.4750481

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4593995, upper bound: 56.4151277
time: 0.57 seconds

## Relational analysis of NS_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4593995, upper bound: 56.4167407
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -126.5899277, 225.5180664, -139.8852844, 255.8192596, -382.4091797, 365.4032593
1: -24.5684433, 28.8065701, -28.1629448, 32.3608208, -56.9292641, 56.9695129
2: -17.7349644, 29.6324329, -20.0881538, 33.1676636, -50.9026222, 49.7205887
3: -16.7349968, 49.4946289, -18.9853401, 55.5363846, -72.2713776, 68.4799576
4: -14.4265404, 37.1043396, -16.3765030, 41.2219124, -55.6484528, 53.4808426

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4430682, upper bound: 56.4072236
time: 0.58 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4395984, upper bound: 56.4074566
time: 0.56 seconds

## BFS NS instance: NS_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -111.9576874, 200.1994019, -92.4740906, 159.1699371, -271.1276245, 292.6734924
1: -21.7531662, 25.3652458, -17.2772198, 20.2644234, -42.0175896, 42.6424637
2: -15.7027578, 26.1739788, -12.6089411, 21.1121616, -36.8149185, 38.7829208
3: -14.7932520, 44.0467644, -12.1863928, 35.5379410, -50.3311920, 56.2331505
4: -12.7256889, 32.7263641, -10.2946796, 26.3329124, -39.0586014, 43.0210419

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1_B1_A1

### Relational analysis result of NS_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4524562, upper bound: 56.4313140
time: 0.69 seconds

## Relational analysis of NS_A1_A2_A1_B1_A2

### Relational analysis result of NS_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4524562, upper bound: 56.4314296
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -117.6332321, 211.6254883, -146.6238098, 269.6494141, -387.2826538, 358.2492981
1: -23.0491028, 26.8075771, -29.7012501, 34.1082649, -57.1573639, 56.5088158
2: -16.5922089, 27.6461983, -21.1765976, 34.9507256, -51.5429306, 48.8227959
3: -15.6193447, 46.4240494, -20.0556469, 58.4634247, -74.0827713, 66.4796906
4: -13.4649029, 34.5149879, -17.2967682, 43.3904076, -56.8553085, 51.8117561

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4428638, upper bound: 56.4335009
time: 0.70 seconds

## Relational analysis of NS_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4428638, upper bound: 56.4343720
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -116.7456284, 206.8684540, -92.7111893, 159.6936188, -276.4392090, 299.5796509
1: -22.5887737, 26.4389706, -17.3367367, 20.3295593, -42.9183311, 43.7757034
2: -16.3748722, 27.0957298, -12.6488457, 21.1737022, -37.5485725, 39.7445717
3: -15.5432720, 45.5241623, -12.2212906, 35.6425056, -51.1857719, 57.7454529
4: -13.3319912, 33.7827110, -10.3274632, 26.4092655, -39.7412567, 44.1101761

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4515402, upper bound: 56.4173736
time: 0.60 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4530815, upper bound: 56.4222904
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -123.0780029, 219.5582428, -146.8838806, 270.1979675, -393.2759705, 366.4421082
1: -23.9895897, 28.0217838, -29.7643852, 34.1768074, -58.1663971, 57.7861710
2: -17.3512402, 28.7338657, -21.2187729, 35.0179520, -52.3691940, 49.9526367
3: -16.4570351, 48.1956482, -20.0942192, 58.5754318, -75.0324554, 68.2898560
4: -14.1394176, 35.8214836, -17.3316307, 43.4731331, -57.6125412, 53.1531105

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4387140, upper bound: 56.4214658
time: 0.60 seconds

## Relational analysis of NS_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4387140, upper bound: 56.4214658
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -139.9903107, 256.0911560, -146.7868042, 271.2817078, -411.2720032, 402.8779602
1: -28.2039471, 32.3871460, -29.8981628, 34.2219620, -62.4259109, 62.2853088
2: -20.1092567, 33.2043266, -21.2400036, 35.1723442, -55.2816010, 54.4443283
3: -19.0101109, 55.5945129, -20.0754395, 58.7820320, -77.7921448, 75.6699448
4: -16.3945637, 41.2546387, -17.3301086, 43.6437874, -60.0383453, 58.5847435

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4408929, upper bound: 56.4476768
time: 0.61 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4428905
time: 0.60 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4509004
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -146.4367065, 267.9514771, -146.7868042, 271.2817078, -417.7184143, 414.7382812
1: -29.4802933, 33.9294548, -29.8981628, 34.2219620, -63.7022552, 63.8276138
2: -21.0296383, 34.7687798, -21.2400036, 35.1723442, -56.2019806, 56.0087814
3: -19.8944378, 58.1176987, -20.0754395, 58.7820320, -78.6764679, 78.1931305
4: -17.1570969, 43.2151527, -17.3301086, 43.6437874, -60.8008842, 60.5452576

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4408929, upper bound: 56.4476768
time: 0.62 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4428905
time: 0.58 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4509004
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -135.6018524, 247.4037628, -118.8304672, 214.5305634, -350.1324158, 366.2342224
1: -27.2749138, 31.2933769, -23.5486107, 27.1795464, -54.4544563, 54.8419876
2: -19.4551163, 32.0965347, -16.8724155, 28.1091499, -47.5642548, 48.9689407
3: -18.4206181, 53.7995682, -16.0914078, 46.9843788, -65.4049911, 69.8909760
4: -15.8637838, 39.8495674, -13.7840042, 34.9143524, -50.7781372, 53.6335716

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_B1_B1

### Relational analysis result of NS_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4051780, upper bound: 56.4049883
time: 0.62 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2

### Relational analysis result of NS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4156665, upper bound: 56.4396046
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -135.4950409, 246.7794952, -277.8045044, 520.1121216, -655.6071777, 524.5839844
1: -27.1509094, 31.2462425, -58.2170296, 66.0261002, -93.1770020, 89.4632721
2: -19.4019833, 32.0328484, -40.8698006, 67.5385742, -86.9405518, 72.9026489
3: -18.3582153, 53.6602516, -38.9033051, 112.4720840, -130.8302917, 92.5635529
4: -15.8114090, 39.8202553, -33.6789360, 83.4079361, -99.2193375, 73.4991913

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4671773, upper bound: 56.4498446
time: 0.66 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4514060, upper bound: 56.4067327
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -139.9903107, 256.0911560, -153.5695343, 283.6153564, -423.6056519, 409.6606750
1: -28.2039471, 32.3871460, -31.2477360, 35.8313904, -64.0353394, 63.6348801
2: -20.1092567, 33.2043266, -22.2131824, 36.8044777, -56.9137306, 55.4175034
3: -19.0101109, 55.5945129, -20.9997807, 61.4355774, -80.4456863, 76.5942917
4: -16.3945637, 41.2546387, -18.1305389, 45.6889343, -62.0834923, 59.3851738

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4383607, upper bound: 56.4475430
time: 0.64 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4399988
time: 0.57 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4480086
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -146.4367065, 267.9514771, -153.5695343, 283.6153564, -430.0520630, 421.5209656
1: -29.4802933, 33.9294548, -31.2477360, 35.8313904, -65.3116760, 65.1771698
2: -21.0296383, 34.7687798, -22.2131824, 36.8044777, -57.8341141, 56.9819565
3: -19.8944378, 58.1176987, -20.9997807, 61.4355774, -81.3300171, 79.1174774
4: -17.1570969, 43.2151527, -18.1305389, 45.6889343, -62.8460312, 61.3456917

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4383607, upper bound: 56.4475429
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4399988
time: 0.63 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4480086
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -139.9903107, 256.0911560, -155.4292450, 286.1744995, -426.1647949, 411.5203552
1: -28.2039471, 32.3871460, -31.5192776, 36.2066917, -64.4106369, 63.9064255
2: -20.1092567, 33.2043266, -22.4438629, 37.1029854, -57.2122421, 55.6481857
3: -19.0101109, 55.5945129, -21.2429676, 61.9818573, -80.9919662, 76.8374710
4: -16.3945637, 41.2546387, -18.3347549, 46.0759964, -62.4705544, 59.5893898

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4411746
time: 0.58 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4487649
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -146.4367065, 267.9514771, -155.4292450, 286.1744995, -432.6112061, 423.3806458
1: -29.4802933, 33.9294548, -31.5192776, 36.2066917, -65.6869812, 65.4487305
2: -21.0296383, 34.7687798, -22.4438629, 37.1029854, -58.1326218, 57.2126427
3: -19.8944378, 58.1176987, -21.2429676, 61.9818573, -81.8762970, 79.3606567
4: -17.1570969, 43.2151527, -18.3347549, 46.0759964, -63.2330933, 61.5499039

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4411746
time: 0.60 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4487649
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.69 seconds
NS_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4423234, upper bound: 56.4139676
NS_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4423234, upper bound: 56.4155910
NS_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4441838, upper bound: 56.4245651
NS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4370419, upper bound: 56.4073689
NS_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4593995, upper bound: 56.4151277
NS_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4593995, upper bound: 56.4167407
NS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4430682, upper bound: 56.4072236
NS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4395984, upper bound: 56.4074566
NS_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4524562, upper bound: 56.4313140
NS_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4524562, upper bound: 56.4314296
NS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4428638, upper bound: 56.4335009
NS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4428638, upper bound: 56.4343720
NS_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4515402, upper bound: 56.4173736
NS_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4530815, upper bound: 56.4222904
NS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4387140, upper bound: 56.4214658
NS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4387140, upper bound: 56.4214658
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4428905
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4509004
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4428905
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4644866, upper bound: 56.4509004
NS_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4051780, upper bound: 56.4049883
NS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4156665, upper bound: 56.4396046
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4671773, upper bound: 56.4498446
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4514060, upper bound: 56.4067327
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4399988
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4480086
NS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4399988
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4480086
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4411746
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4487649
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4411746
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 4, lower bound: -56.4399988, upper bound: 56.4487649

## BFS NS instance: NS_A1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -131.8846130, 232.3807068, -136.2755585, 246.2902679, -378.1747742, 368.6562195
1: -25.1723557, 29.8156910, -26.9791889, 31.3460903, -56.5184441, 56.7948799
2: -18.3670311, 30.5672283, -19.4025211, 32.1306343, -50.4976616, 49.9697380
3: -17.3217754, 51.1863861, -18.3838043, 53.7354774, -71.0572510, 69.5701904
4: -14.9206762, 38.2823868, -15.8388062, 39.9675217, -54.8881912, 54.1211929

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4198502, upper bound: 56.4112683
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_A1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4198502, upper bound: 56.4139676
time: 0.55 seconds

## BFS NS instance: NS_A1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -131.2823944, 230.7921753, -136.2755585, 246.2902679, -377.5726624, 367.0677490
1: -25.0163193, 29.6287003, -26.9791889, 31.3460903, -56.3624077, 56.6078873
2: -18.2530479, 30.3810101, -19.4025211, 32.1306343, -50.3836823, 49.7835236
3: -17.2375641, 50.8810806, -18.3838043, 53.7354774, -70.9730377, 69.2648773
4: -14.8366480, 38.0404282, -15.8388062, 39.9675217, -54.8041687, 53.8792343

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_A1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4198502, upper bound: 56.4128437
time: 0.63 seconds

## Relational analysis of NS_A1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_A1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4198502, upper bound: 56.4155910
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -120.2899933, 211.1528015, -136.2755585, 246.2902679, -366.5802612, 347.4283447
1: -22.7622223, 27.0773335, -26.9791889, 31.3460903, -54.1083145, 54.0565224
2: -16.6596565, 27.9337635, -19.4025211, 32.1306343, -48.7902908, 47.3362808
3: -15.7068615, 46.6762886, -18.3838043, 53.7354774, -69.4423370, 65.0600891
4: -13.4999075, 35.0247383, -15.8388062, 39.9675217, -53.4674301, 50.8635445

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4370419, upper bound: 56.4067075
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4370419, upper bound: 56.4073689
time: 0.55 seconds

## BFS NS instance: NS_A1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -117.4563980, 207.0981598, -136.2755585, 246.2902679, -363.7466431, 343.3737183
1: -22.4420967, 26.5449238, -26.9791889, 31.3460903, -53.7881851, 53.5241127
2: -16.3522606, 27.2924690, -19.4025211, 32.1306343, -48.4828949, 46.6949844
3: -15.4186983, 45.6582375, -18.3838043, 53.7354774, -69.1541595, 64.0420380
4: -13.2701912, 34.2022514, -15.8388062, 39.9675217, -53.2377090, 50.0410576

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4370419, upper bound: 56.4067075
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4370419, upper bound: 56.4073689
time: 0.53 seconds

## BFS NS instance: NS_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -131.8846130, 232.3807068, -139.8852844, 255.8192596, -387.7038269, 372.2658386
1: -25.1723557, 29.8156910, -28.1629448, 32.3608208, -57.5331726, 57.9786377
2: -18.3670311, 30.5672283, -20.0881538, 33.1676636, -51.5346832, 50.6553802
3: -17.3217754, 51.1863861, -18.9853401, 55.5363846, -72.8581619, 70.1717224
4: -14.9206762, 38.2823868, -16.3765030, 41.2219124, -56.1425896, 54.6588860

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4548757, upper bound: 56.4130900
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4548757, upper bound: 56.4151277
time: 0.57 seconds

## BFS NS instance: NS_A1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -131.2823944, 230.7921753, -139.8852844, 255.8192596, -387.1016541, 370.6774292
1: -25.0163193, 29.6287003, -28.1629448, 32.3608208, -57.3771362, 57.7916451
2: -18.2530479, 30.3810101, -20.0881538, 33.1676636, -51.4207115, 50.4691620
3: -17.2375641, 50.8810806, -18.9853401, 55.5363846, -72.7739487, 69.8664246
4: -14.8366480, 38.0404282, -16.3765030, 41.2219124, -56.0585594, 54.4169312

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4552776, upper bound: 56.4145774
time: 0.58 seconds

## Relational analysis of NS_A1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4552776, upper bound: 56.4167407
time: 3.95 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -126.5899277, 225.5180664, -136.7225189, 247.8630219, -374.4529419, 362.2406006
1: -24.5684433, 28.8065701, -27.1788921, 31.4337101, -56.0021515, 55.9854622
2: -17.7349644, 29.6324329, -19.4823380, 32.2873726, -50.0223389, 49.1147690
3: -16.7349968, 49.4946289, -18.4178963, 54.0100174, -70.7450104, 67.9125061
4: -14.4265404, 37.1043396, -15.8598452, 40.1691742, -54.5957146, 52.9641838

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4395984, upper bound: 56.4072236
time: 0.57 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4395984, upper bound: 56.4072236
time: 0.55 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -126.5899277, 225.5180664, -130.8002319, 237.6291809, -364.2191162, 356.3182983
1: -24.5684433, 28.8065701, -26.1589031, 30.1168098, -54.6852531, 54.9654732
2: -17.7349644, 29.6324329, -18.7075901, 30.8604221, -48.5953827, 48.3400154
3: -16.7349968, 49.4946289, -17.6992054, 51.7090149, -68.4440002, 67.1938324
4: -14.4265404, 37.1043396, -15.2403307, 38.3613815, -52.7879219, 52.3446693

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4395984, upper bound: 56.4074566
time: 0.62 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4395984, upper bound: 56.4074566
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -110.6267242, 196.8687134, -92.4740906, 159.1699371, -269.7966614, 289.3428040
1: -21.2939949, 24.9482651, -17.2772198, 20.2644234, -41.5584145, 42.2254868
2: -15.4249535, 25.8066006, -12.6089411, 21.1121616, -36.5371170, 38.4155350
3: -14.5106554, 43.4258270, -12.1863928, 35.5379410, -50.0485916, 55.6122169
4: -12.4719143, 32.2932587, -10.2946796, 26.3329124, -38.8048248, 42.5879288

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4221416, upper bound: 56.4283157
time: 0.65 seconds

## Relational analysis of NS_A1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4202342, upper bound: 56.4313140
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -106.2292252, 188.8295746, -92.4740906, 159.1699371, -265.3991699, 281.3036499
1: -20.5057316, 23.9410763, -17.2772198, 20.2644234, -40.7701569, 41.2182961
2: -14.8252048, 24.7043362, -12.6089411, 21.1121616, -35.9373665, 37.3132706
3: -13.9778004, 41.6640549, -12.1863928, 35.5379410, -49.5157394, 53.8504448
4: -12.0004845, 30.9193306, -10.2946796, 26.3329124, -38.3333931, 41.2140083

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4225511, upper bound: 56.4284040
time: 0.70 seconds

## Relational analysis of NS_A1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4225511, upper bound: 56.4314296
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -117.6332321, 211.6254883, -133.4781189, 246.0440216, -363.6772156, 345.1036072
1: -23.0491028, 26.8075771, -27.0300941, 30.9315586, -53.9806519, 53.8376656
2: -16.5922089, 27.6461983, -19.2012539, 31.8754520, -48.4676590, 46.8474464
3: -15.6193447, 46.4240494, -18.1002407, 53.3359070, -68.9552460, 64.5242920
4: -13.4649029, 34.5149879, -15.6285295, 39.6454544, -53.1103516, 50.1435165

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4212823, upper bound: 56.4291675
time: 0.67 seconds

## Relational analysis of NS_A1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4212823, upper bound: 56.4335009
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -117.6332321, 211.6254883, -138.9709320, 253.5427246, -371.1759033, 350.5964355
1: -23.0491028, 26.8075771, -27.9001751, 32.0833588, -55.1324615, 54.7077408
2: -16.5922089, 27.6461983, -19.9301567, 32.9080391, -49.5002441, 47.5763474
3: -15.6193447, 46.4240494, -18.8921013, 55.0857773, -70.7051239, 65.3161469
4: -13.4649029, 34.5149879, -16.2679462, 40.8921471, -54.3570480, 50.7829361

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4384411, upper bound: 56.4343720
time: 0.68 seconds

## Relational analysis of NS_A1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4427162, upper bound: 56.4310563
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -112.3688049, 196.6710663, -92.7111893, 159.6936188, -272.0623779, 289.3822327
1: -21.3941154, 25.1818295, -17.3367367, 20.3295593, -41.7236557, 42.5185585
2: -15.5933161, 25.8827286, -12.6488457, 21.1737022, -36.7670135, 38.5315742
3: -14.8067484, 43.5294075, -12.2212906, 35.6425056, -50.4492531, 55.7506943
4: -12.6685410, 32.3097610, -10.3274632, 26.4092655, -39.0778046, 42.6372223

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4499917, upper bound: 56.4163774
time: 0.56 seconds

## Relational analysis of NS_A1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4499917, upper bound: 56.4173736
time: 0.52 seconds

## BFS NS instance: NS_A1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -109.9884796, 193.1923218, -92.7111893, 159.6936188, -269.6820679, 285.9035034
1: -21.1034050, 24.7178707, -17.3367367, 20.3295593, -41.4329491, 42.0545921
2: -15.3330441, 25.3444462, -12.6488457, 21.1737022, -36.5067368, 37.9932899
3: -14.5908413, 42.6942368, -12.2212906, 35.6425056, -50.2333412, 54.9155273
4: -12.4784021, 31.6013546, -10.3274632, 26.4092655, -38.8876648, 41.9288139

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4530815, upper bound: 56.4222904
time: 0.63 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4514921, upper bound: 56.4210333
time: 0.56 seconds

## Relational analysis of NS_A1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4514921, upper bound: 56.4222904
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -123.0780029, 219.5582428, -133.4781189, 246.0440216, -369.1219788, 353.0363770
1: -23.9895897, 28.0217838, -27.0300941, 30.9315586, -54.9211502, 55.0518799
2: -17.3512402, 28.7338657, -19.2012539, 31.8754520, -49.2266922, 47.9351196
3: -16.4570351, 48.1956482, -18.1002407, 53.3359070, -69.7929153, 66.2958679
4: -14.1394176, 35.8214836, -15.6285295, 39.6454544, -53.7848625, 51.4500122

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4171324, upper bound: 56.4171324
time: 0.57 seconds

## Relational analysis of NS_A1_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4171324, upper bound: 56.4214658
time: 0.56 seconds

## BFS NS instance: NS_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -123.0780029, 219.5582428, -138.9709320, 253.5427246, -376.6206970, 358.5291748
1: -23.9895897, 28.0217838, -27.9001751, 32.0833588, -56.0729485, 55.9219589
2: -17.3512402, 28.7338657, -19.9301567, 32.9080391, -50.2592773, 48.6640205
3: -16.4570351, 48.1956482, -18.8921013, 55.0857773, -71.5427933, 67.0877533
4: -14.1394176, 35.8214836, -16.2679462, 40.8921471, -55.0315590, 52.0894318

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4171324, upper bound: 56.4171324
time: 0.55 seconds

## Relational analysis of NS_A1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4171324, upper bound: 56.4214658
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -136.7176666, 250.2311707, -146.7868042, 271.2817078, -407.9993896, 397.0179749
1: -27.5542049, 31.6117134, -29.8981628, 34.2219620, -61.7761650, 61.5098686
2: -19.6216354, 32.4831696, -21.2400036, 35.1723442, -54.7939796, 53.7231750
3: -18.5524902, 54.3575668, -20.0754395, 58.7820320, -77.3345108, 74.4329910
4: -15.9856529, 40.3501778, -17.3301086, 43.6437874, -59.6294403, 57.6802826

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -139.0081635, 253.9446869, -146.7868042, 271.2817078, -410.2898254, 400.7315063
1: -27.9569759, 32.1241646, -29.8981628, 34.2219620, -62.1789360, 62.0223160
2: -19.9492397, 32.9407120, -21.2400036, 35.1723442, -55.1215820, 54.1807175
3: -18.8669243, 55.1621170, -20.0754395, 58.7820320, -77.6489563, 75.2375565
4: -16.2635612, 40.9298401, -17.3301086, 43.6437874, -59.9073448, 58.2599449

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -143.5615082, 262.8895569, -146.7868042, 271.2817078, -414.8432007, 409.6763611
1: -28.9328880, 33.2466278, -29.8981628, 34.2219620, -63.1548500, 63.1447906
2: -20.6064663, 34.1516075, -21.2400036, 35.1723442, -55.7788086, 55.3916092
3: -19.4946423, 57.0675125, -20.0754395, 58.7820320, -78.2766724, 77.1429520
4: -16.7995949, 42.4331818, -17.3301086, 43.6437874, -60.4433784, 59.7632828

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -145.4452362, 265.7645874, -146.7868042, 271.2817078, -416.7269287, 412.5513916
1: -29.2288094, 33.6603813, -29.8981628, 34.2219620, -63.4507713, 63.5585327
2: -20.8657475, 34.5036011, -21.2400036, 35.1723442, -56.0380936, 55.7436066
3: -19.7484131, 57.6790276, -20.0754395, 58.7820320, -78.5304413, 77.7544556
4: -17.0230331, 42.8868065, -17.3301086, 43.6437874, -60.6668205, 60.2169075

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -135.6018524, 247.4037628, -110.0551758, 196.2556458, -331.8574829, 357.4589233
1: -27.2749138, 31.2933769, -21.5314445, 24.8577480, -52.1326599, 52.8248215
2: -19.4551163, 32.0965347, -15.4714890, 25.7713871, -45.2265015, 47.5680161
3: -18.4206181, 53.7995682, -14.8173618, 43.1887093, -61.6093292, 68.6169281
4: -15.8637838, 39.8495674, -12.6260519, 32.0499191, -47.9137039, 52.4756203

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -127.4662094, 229.1123810, -276.2086792, 516.9780273, -644.4442139, 505.3209839
1: -25.1537514, 29.0761166, -57.8549576, 65.6329575, -90.7867126, 86.9310608
2: -18.0785885, 29.8441219, -40.6225967, 67.1436615, -85.2222443, 70.4667130
3: -17.1804867, 50.0559273, -38.6653786, 111.8100052, -128.9904938, 88.7212830
4: -14.7415771, 37.1195030, -33.4729881, 82.9235229, -97.6651001, 70.5924911

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4176710, upper bound: 56.4461477
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4641676, upper bound: 56.4487749
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -134.2839966, 244.4000854, -277.8045044, 520.1121216, -654.3960571, 522.2045898
1: -26.8858795, 30.9517746, -58.2170296, 66.0261002, -92.9119644, 89.1688080
2: -19.2124081, 31.7291374, -40.8698006, 67.5385742, -86.7509689, 72.5989304
3: -18.1718121, 53.1567726, -38.9033051, 112.4720840, -130.6438904, 92.0600662
4: -15.6508608, 39.4518280, -33.6789360, 83.4079361, -99.0587997, 73.1307678

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -56.4061735, upper bound: 56.3972458
time: 0.60 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -56.4474508, upper bound: 56.4043899
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -136.7176666, 250.2311707, -153.5695343, 283.6153564, -420.3330078, 403.8006592
1: -27.5542049, 31.6117134, -31.2477360, 35.8313904, -63.3855972, 62.8594284
2: -19.6216354, 32.4831696, -22.2131824, 36.8044777, -56.4261093, 54.6963425
3: -18.5524902, 54.3575668, -20.9997807, 61.4355774, -79.9880600, 75.3573456
4: -15.9856529, 40.3501778, -18.1305389, 45.6889343, -61.6745872, 58.4807129

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -139.0081635, 253.9446869, -153.5695343, 283.6153564, -422.6234741, 407.5141602
1: -27.9569759, 32.1241646, -31.2477360, 35.8313904, -63.7883682, 63.3718796
2: -19.9492397, 32.9407120, -22.2131824, 36.8044777, -56.7537079, 55.1538887
3: -18.8669243, 55.1621170, -20.9997807, 61.4355774, -80.3025055, 76.1618958
4: -16.2635612, 40.9298401, -18.1305389, 45.6889343, -61.9524918, 59.0603790

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -143.5615082, 262.8895569, -153.5695343, 283.6153564, -427.1768799, 416.4590454
1: -28.9328880, 33.2466278, -31.2477360, 35.8313904, -64.7642670, 64.4943390
2: -20.6064663, 34.1516075, -22.2131824, 36.8044777, -57.4109421, 56.3647842
3: -19.4946423, 57.0675125, -20.9997807, 61.4355774, -80.9302216, 78.0672913
4: -16.7995949, 42.4331818, -18.1305389, 45.6889343, -62.4885254, 60.5637169

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -145.4452362, 265.7645874, -153.5695343, 283.6153564, -429.0605774, 419.3341064
1: -29.2288094, 33.6603813, -31.2477360, 35.8313904, -65.0601883, 64.9080811
2: -20.8657475, 34.5036011, -22.2131824, 36.8044777, -57.6702156, 56.7167816
3: -19.7484131, 57.6790276, -20.9997807, 61.4355774, -81.1839905, 78.6788101
4: -17.0230331, 42.8868065, -18.1305389, 45.6889343, -62.7119675, 61.0173378

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -136.7176666, 250.2311707, -155.4292450, 286.1744995, -422.8921509, 405.6603699
1: -27.5542049, 31.6117134, -31.5192776, 36.2066917, -63.7608948, 63.1309853
2: -19.6216354, 32.4831696, -22.4438629, 37.1029854, -56.7246208, 54.9270287
3: -18.5524902, 54.3575668, -21.2429676, 61.9818573, -80.5343399, 75.6005249
4: -15.9856529, 40.3501778, -18.3347549, 46.0759964, -62.0616493, 58.6849289

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -139.0081635, 253.9446869, -155.4292450, 286.1744995, -425.1826172, 409.3738403
1: -27.9569759, 32.1241646, -31.5192776, 36.2066917, -64.1636658, 63.6434402
2: -19.9492397, 32.9407120, -22.4438629, 37.1029854, -57.0522194, 55.3845711
3: -18.8669243, 55.1621170, -21.2429676, 61.9818573, -80.8487854, 76.4050827
4: -16.2635612, 40.9298401, -18.3347549, 46.0759964, -62.3395538, 59.2645912

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -143.5615082, 262.8895569, -155.4292450, 286.1744995, -429.7359924, 418.3187561
1: -28.9328880, 33.2466278, -31.5192776, 36.2066917, -65.1395798, 64.7659073
2: -20.6064663, 34.1516075, -22.4438629, 37.1029854, -57.7094498, 56.5954666
3: -19.4946423, 57.0675125, -21.2429676, 61.9818573, -81.4765015, 78.3104782
4: -16.7995949, 42.4331818, -18.3347549, 46.0759964, -62.8755875, 60.7679291

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -145.4452362, 265.7645874, -155.4292450, 286.1744995, -431.6197205, 421.1937866
1: -29.2288094, 33.6603813, -31.5192776, 36.2066917, -65.4355011, 65.1796494
2: -20.8657475, 34.5036011, -22.4438629, 37.1029854, -57.9687271, 56.9474602
3: -19.7484131, 57.6790276, -21.2429676, 61.9818573, -81.7302704, 78.9219818
4: -17.0230331, 42.8868065, -18.3347549, 46.0759964, -63.0990295, 61.2215538

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.00 + 255.15 = 259.14 seconds
