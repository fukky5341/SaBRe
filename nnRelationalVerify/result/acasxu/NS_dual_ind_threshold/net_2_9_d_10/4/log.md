## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 14.783633487000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007)
1: (-10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563)
2: (-6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616)
3: (-7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103)
4: (-5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.11 + 1.51 = 4.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8206852

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8185006
time: 0.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8206852
time: 0.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8185006
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8206852

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -80.1466370, 36.2084389, -129.1032867, 75.8428421, -155.9894714, 165.3117218
1: -4.2907152, 4.0505705, -8.1762457, 6.5313454, -10.8220606, 12.2268152
2: -3.1247096, 5.3048964, -5.1434026, 10.2446337, -13.3693428, 10.4482985
3: -3.7052639, 9.9149799, -6.5708523, 17.1385098, -20.8437691, 16.4858303
4: -2.5783277, 5.8772559, -4.4006977, 10.6625814, -13.2409077, 10.2779541

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8185006
time: 0.52 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8185006
time: 0.53 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -195.6805420, 144.9261017, -145.9259338, 94.2153473, -289.8958435, 290.8520508
1: -14.6865873, 9.9890881, -9.9135609, 7.4269538, -22.1135406, 19.9026470
2: -8.3289146, 18.2123413, -5.8846374, 12.3311110, -20.6600266, 24.0969791
3: -11.1344271, 27.9955597, -7.6238966, 19.8503265, -30.9847527, 35.6194572
4: -7.0948596, 18.8732567, -5.0697403, 12.8165951, -19.9114552, 23.9429913

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8206852
time: 0.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8206852
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8185006
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8185006
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8206852
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 4, lower bound: -14.8185006, upper bound: 14.8206852

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -80.1466370, 36.2084389, -80.1466370, 36.2084389, -116.3550644, 116.3550568
1: -4.2907152, 4.0505705, -4.2907152, 4.0505705, -8.3412857, 8.3412857
2: -3.1247096, 5.3048964, -3.1247096, 5.3048964, -8.4296055, 8.4296055
3: -3.7052639, 9.9149799, -3.7052639, 9.9149799, -13.6202431, 13.6202431
4: -2.5783277, 5.8772559, -2.5783277, 5.8772559, -8.4555836, 8.4555836

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8125058, upper bound: 14.7961696
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8125058, upper bound: 14.8125058
time: 0.51 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -80.1466370, 36.2084389, -193.5660400, 137.8887482, -218.0353851, 229.7744751
1: -4.2907152, 4.0505705, -14.0066881, 9.8857279, -14.1764431, 18.0572586
2: -3.1247096, 5.3048964, -8.1867943, 17.7747803, -20.8994884, 13.4916897
3: -3.7052639, 9.9149799, -10.9608564, 27.4590988, -31.1643562, 20.8758316
4: -2.5783277, 5.8772559, -7.0262671, 18.3796406, -20.9579639, 12.9035225

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8125058, upper bound: 14.7961696
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8125058, upper bound: 14.8125058
time: 0.57 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -195.6805420, 144.9261017, -80.1466370, 36.2084389, -231.8889771, 225.0727386
1: -14.6865873, 9.9890881, -4.2907152, 4.0505705, -18.7371559, 14.2798033
2: -8.3289146, 18.2123413, -3.1247096, 5.3048964, -13.6338110, 21.3370514
3: -11.1344271, 27.9955597, -3.7052639, 9.9149799, -21.0494061, 31.7008152
4: -7.0948596, 18.8732567, -2.5783277, 5.8772559, -12.9721155, 21.4515839

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8194346
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
time: 0.54 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -195.6805420, 144.9261017, -195.1378479, 144.4280548, -340.1085815, 340.0639648
1: -14.6865873, 9.9890881, -14.6371574, 9.9643288, -24.6509151, 24.6262455
2: -8.3289146, 18.2123413, -8.3041458, 18.1441193, -26.4730339, 26.5164871
3: -11.1344271, 27.9955597, -11.0937767, 27.9057598, -39.0401840, 39.0893364
4: -7.0948596, 18.8732567, -7.0717521, 18.8060417, -25.9009018, 25.9450092

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8195538
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.13 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.8125058, upper bound: 14.7961696
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.8125058, upper bound: 14.8125058
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.8125058, upper bound: 14.7961696
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.8125058, upper bound: 14.8125058
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8194346
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8195538
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -75.2906036, 31.3227730, -80.1466370, 36.2084389, -111.4990311, 111.4694061
1: -3.8496413, 3.8229699, -4.2907152, 4.0505705, -7.9002118, 8.1136856
2: -2.8791416, 4.6194034, -3.1247096, 5.3048964, -8.1840382, 7.7441125
3: -3.3371439, 9.0236359, -3.7052639, 9.9149799, -13.2521238, 12.7288990
4: -2.3911729, 5.1706152, -2.5783277, 5.8772559, -8.2684278, 7.7489429

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7961696, upper bound: 14.7961696
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7961696, upper bound: 14.7961696
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -76.6764984, 33.5508270, -80.1466370, 36.2084389, -112.8849335, 113.6974640
1: -4.0355144, 3.8843422, -4.2907152, 4.0505705, -8.0860844, 8.1750574
2: -2.9608030, 4.9485660, -3.1247096, 5.3048964, -8.2656994, 8.0732756
3: -3.4688039, 9.3769484, -3.7052639, 9.9149799, -13.3837833, 13.0822105
4: -2.4528494, 5.5034761, -2.5783277, 5.8772559, -8.3301048, 8.0818043

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7961696, upper bound: 14.8125058
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7961696, upper bound: 14.8125058
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -75.2906036, 31.3227730, -193.5660400, 137.8887482, -213.1793518, 224.8888092
1: -3.8496413, 3.8229699, -14.0066881, 9.8857279, -13.7353687, 17.8296585
2: -2.8791416, 4.6194034, -8.1867943, 17.7747803, -20.6539192, 12.8061962
3: -3.3371439, 9.0236359, -10.9608564, 27.4590988, -30.7962418, 19.9844856
4: -2.3911729, 5.1706152, -7.0262671, 18.3796406, -20.7708073, 12.1968813

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8101290, upper bound: 14.7925923
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7925923
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -76.6764984, 33.5508270, -193.5660400, 137.8887482, -214.5652466, 227.1168671
1: -4.0355144, 3.8843422, -14.0066881, 9.8857279, -13.9212418, 17.8910294
2: -2.9608030, 4.9485660, -8.1867943, 17.7747803, -20.7355804, 13.1353607
3: -3.4688039, 9.3769484, -10.9608564, 27.4590988, -30.9279022, 20.3378029
4: -2.4528494, 5.5034761, -7.0262671, 18.3796406, -20.8324871, 12.5297432

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8141343, upper bound: 14.7707386
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8063746, upper bound: 14.7707386
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -179.4290466, 126.6083527, -80.1466370, 36.2084389, -215.6374817, 206.7549896
1: -12.9703083, 9.1483183, -4.2907152, 4.0505705, -17.0208778, 13.4390335
2: -7.5566487, 16.1841888, -3.1247096, 5.3048964, -12.8615446, 19.3088970
3: -10.0429697, 25.3084297, -3.7052639, 9.9149799, -19.9579506, 29.0136890
4: -6.4515276, 16.8023033, -2.5783277, 5.8772559, -12.3287830, 19.3806286

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7456357, upper bound: 14.8168922
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7627130, upper bound: 14.8194346
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -176.4409790, 127.2970657, -80.1466370, 36.2084389, -212.6494141, 207.4436951
1: -13.0086794, 8.9960995, -4.2907152, 4.0505705, -17.0592499, 13.2868147
2: -7.4931507, 16.3116360, -3.1247096, 5.3048964, -12.7980471, 19.4363461
3: -10.1010818, 25.1094704, -3.7052639, 9.9149799, -20.0160618, 28.8147278
4: -6.4500360, 16.9278450, -2.5783277, 5.8772559, -12.3272905, 19.5061722

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7456357, upper bound: 14.8117510
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7627130, upper bound: 14.8127826
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -179.4290466, 126.6083527, -195.1378479, 144.4280548, -323.8571167, 321.7462158
1: -12.9703083, 9.1483183, -14.6371574, 9.9643288, -22.9346371, 23.7854748
2: -7.5566487, 16.1841888, -8.3041458, 18.1441193, -25.7007656, 24.4883327
3: -10.0429697, 25.3084297, -11.0937767, 27.9057598, -37.9487305, 36.4022064
4: -6.4515276, 16.8023033, -7.0717521, 18.8060417, -25.2575684, 23.8740520

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -176.4409790, 127.2970657, -195.1378479, 144.4280548, -320.8689575, 322.4349060
1: -13.0086794, 8.9960995, -14.6371574, 9.9643288, -22.9730072, 23.6332569
2: -7.4931507, 16.3116360, -8.3041458, 18.1441193, -25.6372700, 24.6157818
3: -10.1010818, 25.1094704, -11.0937767, 27.9057598, -38.0068436, 36.2032471
4: -6.4500360, 16.9278450, -7.0717521, 18.8060417, -25.2560768, 23.9995975

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.36 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7961696, upper bound: 14.7961696
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7961696, upper bound: 14.7961696
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7961696, upper bound: 14.8125058
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7961696, upper bound: 14.8125058
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.8101290, upper bound: 14.7925923
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7925923
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.8141343, upper bound: 14.7707386
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.8063746, upper bound: 14.7707386
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7456357, upper bound: 14.8168922
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7627130, upper bound: 14.8194346
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7456357, upper bound: 14.8117510
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7627130, upper bound: 14.8127826
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.7776591, upper bound: 14.8127883
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -75.2906036, 31.3227730, -75.2906036, 31.3227730, -106.6133728, 106.6133728
1: -3.8496413, 3.8229699, -3.8496413, 3.8229699, -7.6726112, 7.6726112
2: -2.8791416, 4.6194034, -2.8791416, 4.6194034, -7.4985447, 7.4985447
3: -3.3371439, 9.0236359, -3.3371439, 9.0236359, -12.3607798, 12.3607798
4: -2.3911729, 5.1706152, -2.3911729, 5.1706152, -7.5617881, 7.5617881

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7665546, upper bound: 14.7042002
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7931223, upper bound: 14.7931221
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -75.2906036, 31.3227730, -76.6764984, 33.5508270, -108.8414307, 107.9992676
1: -3.8496413, 3.8229699, -4.0355144, 3.8843422, -7.7339835, 7.8584843
2: -2.8791416, 4.6194034, -2.9608030, 4.9485660, -7.8277073, 7.5802064
3: -3.3371439, 9.0236359, -3.4688039, 9.3769484, -12.7140923, 12.4924393
4: -2.3911729, 5.1706152, -2.4528494, 5.5034761, -7.8946486, 7.6234646

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7665546, upper bound: 14.7042002
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7931223, upper bound: 14.7931221
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -76.6764984, 33.5508270, -75.2906036, 31.3227730, -107.9992676, 108.8414307
1: -4.0355144, 3.8843422, -3.8496413, 3.8229699, -7.8584843, 7.7339835
2: -2.9608030, 4.9485660, -2.8791416, 4.6194034, -7.5802064, 7.8277073
3: -3.4688039, 9.3769484, -3.3371439, 9.0236359, -12.4924393, 12.7140913
4: -2.4528494, 5.5034761, -2.3911729, 5.1706152, -7.6234646, 7.8946481

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7661633, upper bound: 14.7555339
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7931223, upper bound: 14.8107279
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -76.6764984, 33.5508270, -76.6764984, 33.5508270, -110.2273254, 110.2273254
1: -4.0355144, 3.8843422, -4.0355144, 3.8843422, -7.9198565, 7.9198565
2: -2.9608030, 4.9485660, -2.9608030, 4.9485660, -7.9093690, 7.9093690
3: -3.4688039, 9.3769484, -3.4688039, 9.3769484, -12.8457527, 12.8457527
4: -2.4528494, 5.5034761, -2.4528494, 5.5034761, -7.9563255, 7.9563255

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7661633, upper bound: 14.7555340
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7931223, upper bound: 14.8107279
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -75.2906036, 31.3227730, -172.1016388, 116.7892303, -192.0798187, 203.4244080
1: -3.8496413, 3.8229699, -12.0355148, 8.7626028, -12.6122437, 15.8584843
2: -2.8791416, 4.6194034, -7.1693926, 15.2775908, -18.1567326, 11.7887955
3: -3.3371439, 9.0236359, -9.5271931, 24.0247135, -27.3618584, 18.5508213
4: -2.3911729, 5.1706152, -6.1754236, 15.8352919, -18.2264595, 11.3460379

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7871675
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7925923
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -74.8185883, 30.9369850, -195.5269623, 134.9160614, -209.7346344, 226.4639435
1: -3.8140631, 3.8001549, -13.7809181, 9.9742508, -13.7883139, 17.5810738
2: -2.8563361, 4.5711031, -8.1948099, 17.4344120, -20.2907486, 12.7659130
3: -3.3043232, 8.9514933, -10.9012642, 27.4024811, -30.7068043, 19.8527565
4: -2.3728395, 5.1185579, -7.0257225, 18.0557747, -20.4286137, 12.1442804

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7871675
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7925923
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -76.6764984, 33.5508270, -178.1054993, 122.3268967, -199.0033875, 211.6563263
1: -4.0355144, 3.8843422, -12.5641775, 9.0843706, -13.1198845, 16.4485188
2: -2.9608030, 4.9485660, -7.4684496, 15.9306650, -18.8914642, 12.4170151
3: -3.4688039, 9.3769484, -9.9334354, 24.9782906, -28.4470940, 19.3103828
4: -2.4528494, 5.5034761, -6.4094572, 16.5150242, -18.9678726, 11.9129333

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8124091, upper bound: 14.7412703
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8141343, upper bound: 14.7557006
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -76.6764984, 33.5508270, -174.9891815, 122.6572571, -199.3337555, 208.5399780
1: -4.0355144, 3.8843422, -12.5696659, 8.9273005, -12.9628143, 16.4540081
2: -2.9608030, 4.9485660, -7.3980560, 16.0391769, -18.9999771, 12.3466215
3: -3.4688039, 9.3769484, -9.9804430, 24.7559662, -28.2247696, 19.3573914
4: -2.4528494, 5.5034761, -6.4037957, 16.6205502, -19.0733986, 11.9072714

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8054833, upper bound: 14.7412703
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061557, upper bound: 14.7557006
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -167.6352997, 112.1303253, -61.1898651, 24.4817467, -192.1170502, 173.3201599
1: -11.6204720, 8.5177975, -3.0392938, 3.1128674, -14.7333393, 11.5570908
2: -6.9957857, 14.5512466, -2.3409739, 3.6340690, -10.6298542, 16.8922176
3: -9.2034483, 23.2756901, -2.6836782, 7.2653589, -16.4688072, 25.9593678
4: -5.9534841, 15.1184387, -1.9518619, 4.0945668, -10.0480509, 17.0703011

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8161428
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8075303
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -179.4290466, 126.6083527, -77.1478119, 34.7145233, -214.1435699, 203.7561646
1: -12.9703083, 9.1483183, -4.1263742, 3.8999236, -16.8702316, 13.2746916
2: -7.5566487, 16.1841888, -3.0006537, 5.1204267, -12.6770754, 19.1848431
3: -10.0429697, 25.3084297, -3.5615425, 9.5225945, -19.5655632, 28.8699722
4: -6.4515276, 16.8023033, -2.4845543, 5.6786580, -12.1301861, 19.2868538

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185416
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185040
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -165.3578796, 113.1880417, -61.1898651, 24.4817467, -189.8396301, 174.3778839
1: -11.6925449, 8.4011421, -3.0392938, 3.1128674, -14.8054123, 11.4404354
2: -6.9560943, 14.7242155, -2.3409739, 3.6340690, -10.5901632, 17.0651817
3: -9.2839661, 23.1606903, -2.6836782, 7.2653589, -16.5493240, 25.8443680
4: -5.9732504, 15.2836418, -1.9518619, 4.0945668, -10.0678158, 17.2355003

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8117510
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8076837
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -176.4409790, 127.2970657, -77.1478119, 34.7145233, -211.1555023, 204.4448700
1: -13.0086794, 8.9960995, -4.1263742, 3.8999236, -16.9086037, 13.1224728
2: -7.4931507, 16.3116360, -3.0006537, 5.1204267, -12.6135769, 19.3122902
3: -10.1010818, 25.1094704, -3.5615425, 9.5225945, -19.6236763, 28.6710091
4: -6.4500360, 16.9278450, -2.4845543, 5.6786580, -12.1286926, 19.4123993

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8127826
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8121202
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -179.4290466, 126.6083527, -178.9485779, 126.1504974, -305.5795288, 305.5569458
1: -12.9703083, 9.1483183, -12.9268885, 9.1260834, -22.0963917, 22.0752068
2: -7.5566487, 16.1841888, -7.5344663, 16.1233349, -23.6799812, 23.7186546
3: -10.0429697, 25.3084297, -10.0063629, 25.2291126, -35.2720833, 35.3147888
4: -6.4515276, 16.8023033, -6.4307871, 16.7431602, -23.1946869, 23.2330837

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7707387, upper bound: 14.8012039
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8063746, upper bound: 14.8142218
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -179.4290466, 126.6083527, -175.9706421, 126.8217239, -306.2507019, 302.5789795
1: -12.9703083, 9.1483183, -12.9638186, 8.9737997, -21.9441071, 22.1121349
2: -7.5566487, 16.1841888, -7.4703002, 16.2491837, -23.8058319, 23.6544838
3: -10.0429697, 25.3084297, -10.0635452, 25.0271416, -35.0701103, 35.3719711
4: -6.4515276, 16.8023033, -6.4292040, 16.8677425, -23.3192711, 23.2315006

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7707387, upper bound: 14.8012039
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8063746, upper bound: 14.8142218
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -176.4409790, 127.2970657, -178.9485779, 126.1504974, -302.5914001, 306.2456360
1: -13.0086794, 8.9960995, -12.9268885, 9.1260834, -22.1347618, 21.9229889
2: -7.4931507, 16.3116360, -7.5344663, 16.1233349, -23.6164856, 23.8461018
3: -10.1010818, 25.1094704, -10.0063629, 25.2291126, -35.3301926, 35.1158333
4: -6.4500360, 16.9278450, -6.4307871, 16.7431602, -23.1931953, 23.3586311

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127883
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7724754, upper bound: 14.8127474
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -176.4409790, 127.2970657, -175.9706421, 126.8217239, -303.2625732, 303.2677002
1: -13.0086794, 8.9960995, -12.9638186, 8.9737997, -21.9824772, 21.9599190
2: -7.4931507, 16.3116360, -7.4703002, 16.2491837, -23.7423344, 23.7819328
3: -10.1010818, 25.1094704, -10.0635452, 25.0271416, -35.1282234, 35.1730156
4: -6.4500360, 16.9278450, -6.4292040, 16.8677425, -23.3177776, 23.3570480

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7724754, upper bound: 14.8127883
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127474
time: 0.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.84 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7665546, upper bound: 14.7042002
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7931223, upper bound: 14.7931221
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7665546, upper bound: 14.7042002
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7931223, upper bound: 14.7931221
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7661633, upper bound: 14.7555339
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7931223, upper bound: 14.8107279
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7661633, upper bound: 14.7555340
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7931223, upper bound: 14.8107279
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7871675
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7925923
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7871675
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8091038, upper bound: 14.7925923
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8124091, upper bound: 14.7412703
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8141343, upper bound: 14.7557006
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8054833, upper bound: 14.7412703
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8061557, upper bound: 14.7557006
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8161428
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8075303
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185416
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185040
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8117510
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7251595, upper bound: 14.8076837
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8127826
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8121202
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7707387, upper bound: 14.8012039
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8063746, upper bound: 14.8142218
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7707387, upper bound: 14.8012039
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8063746, upper bound: 14.8142218
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127883
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7724754, upper bound: 14.8127474
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.7724754, upper bound: 14.8127883
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127474

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -72.0984116, 29.8062210, -75.2906036, 31.3227730, -103.4211884, 105.0968246
1: -3.6660242, 3.6619985, -3.8496413, 3.8229699, -7.4889941, 7.5116396
2: -2.7508221, 4.4260740, -2.8791416, 4.6194034, -7.3702250, 7.3052158
3: -3.1673064, 8.6350651, -3.3371439, 9.0236359, -12.1909428, 11.9722090
4: -2.2876673, 4.9355497, -2.3911729, 5.1706152, -7.4582825, 7.3267226

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7665546
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7931223
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -72.0984116, 29.8062210, -76.6764984, 33.5508270, -105.6492386, 106.4827194
1: -3.6660242, 3.6619985, -4.0355144, 3.8843422, -7.5503664, 7.6975126
2: -2.7508221, 4.4260740, -2.9608030, 4.9485660, -7.6993880, 7.3868771
3: -3.1673064, 8.6350651, -3.4688039, 9.3769484, -12.5442543, 12.1038685
4: -2.2876673, 4.9355497, -2.4528494, 5.5034761, -7.7911434, 7.3883991

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7555339, upper bound: 14.7661633
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7555339, upper bound: 14.7931223
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -75.2906036, 31.3227730, -105.0950623, 107.5549469
1: -3.8820581, 3.7356277, -3.8496413, 3.8229699, -7.7050281, 7.5852685
2: -2.8422780, 4.7919307, -2.8791416, 4.6194034, -7.4616809, 7.6710720
3: -3.3325341, 9.0222855, -3.3371439, 9.0236359, -12.3561687, 12.3594294
4: -2.3572531, 5.3203726, -2.3911729, 5.1706152, -7.5278683, 7.7115455

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7930910
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7042002, upper bound: 14.8107280
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -76.6764984, 33.5508270, -107.3231125, 108.9408417
1: -3.8820581, 3.7356277, -4.0355144, 3.8843422, -7.7664003, 7.7711415
2: -2.8422780, 4.7919307, -2.9608030, 4.9485660, -7.7908440, 7.7527337
3: -3.3325341, 9.0222855, -3.4688039, 9.3769484, -12.7094822, 12.4910889
4: -2.3572531, 5.3203726, -2.4528494, 5.5034761, -7.8607292, 7.7732220

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7942578
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7555339, upper bound: 14.8107281
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -71.3583679, 28.1626167, -172.1016388, 116.7892303, -188.1475830, 200.2642517
1: -3.5577404, 3.6309237, -12.0355148, 8.7626028, -12.3203430, 15.6664391
2: -2.6802974, 4.2135391, -7.1693926, 15.2775908, -17.9578876, 11.3829317
3: -3.0570858, 8.4137774, -9.5271931, 24.0247135, -27.0817986, 17.9409657
4: -2.2340636, 4.7260466, -6.1754236, 15.8352919, -18.0693550, 10.9014683

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -85.8713150, 35.4057808, -172.1016388, 116.7892303, -202.6605530, 207.5074005
1: -4.3858232, 4.3748875, -12.0355148, 8.7626028, -13.1484261, 16.4104023
2: -3.2892232, 5.2422342, -7.1693926, 15.2775908, -18.5668144, 12.4116249
3: -3.8124676, 10.2811842, -9.5271931, 24.0247135, -27.8371811, 19.8083725
4: -2.7362194, 5.8653440, -6.1754236, 15.8352919, -18.5715046, 12.0407667

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -71.3583679, 28.1626167, -195.5269623, 134.9160614, -206.2744293, 223.6895752
1: -3.5577404, 3.6309237, -13.7809181, 9.9742508, -13.5319910, 17.4118423
2: -2.6802974, 4.2135391, -8.1948099, 17.4344120, -20.1147079, 12.4083481
3: -3.0570858, 8.4137774, -10.9012642, 27.4024811, -30.4595661, 19.3150387
4: -2.2340636, 4.7260466, -7.0257225, 18.0557747, -20.2898369, 11.7517681

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -85.8517609, 35.3826256, -195.5269623, 134.9160614, -220.7678223, 230.9095917
1: -4.3837962, 4.3737221, -13.7809181, 9.9742508, -14.3580475, 18.1546364
2: -3.2880149, 5.2395525, -8.1948099, 17.4344120, -20.7224274, 13.4343596
3: -3.8089497, 10.2776937, -10.9012642, 27.4024811, -31.2114315, 21.1789589
4: -2.7346220, 5.8627987, -7.0257225, 18.0557747, -20.7903976, 12.8885212

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -59.3079529, 23.4528027, -166.9163818, 109.9266739, -169.2346191, 190.3691711
1: -2.9284151, 3.0196221, -11.4132776, 8.4832640, -11.4116783, 14.4328985
2: -2.2548537, 3.4889143, -6.9488406, 14.4225388, -16.6773930, 10.4377546
3: -2.5633979, 7.0197477, -9.1444092, 23.1015377, -25.6649342, 16.1641579
4: -1.8874598, 3.9327717, -5.9308152, 14.9732599, -16.8607197, 9.8635864

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118611, upper bound: 14.7188680
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8007433, upper bound: 14.7188680
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -178.1054993, 122.3268967, -196.0991821, 210.3698425
1: -3.8820581, 3.7356277, -12.5641775, 9.0843706, -12.9664288, 16.2998047
2: -2.8422780, 4.7919307, -7.4684496, 15.9306650, -18.7729416, 12.2603798
3: -3.3325341, 9.0222855, -9.9334354, 24.9782906, -28.3108253, 18.9557209
4: -2.3572531, 5.3203726, -6.4094572, 16.5150242, -18.8722763, 11.7298298

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950703, upper bound: 14.7557006
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950704, upper bound: 14.7557006
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -59.3079529, 23.4528027, -164.5045776, 110.6049957, -169.9129486, 187.9573822
1: -2.9284151, 3.0196221, -11.4471016, 8.3607531, -11.2891684, 14.4667215
2: -2.2548537, 3.4889143, -6.9005880, 14.5718346, -16.8266888, 10.3895025
3: -2.5633979, 7.0197477, -9.2126312, 22.9548759, -25.5182743, 16.2323799
4: -1.8874598, 3.9327717, -5.9460020, 15.1148968, -17.0023537, 9.8787737

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7834507, upper bound: 14.7363767
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7834508, upper bound: 14.7363767
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -174.9891815, 122.6572571, -196.4295502, 207.2535248
1: -3.8820581, 3.7356277, -12.5696659, 8.9273005, -12.8093586, 16.3052940
2: -2.8422780, 4.7919307, -7.3980560, 16.0391769, -18.8814545, 12.1899862
3: -3.3325341, 9.0222855, -9.9804430, 24.7559662, -28.0884991, 19.0027275
4: -2.3572531, 5.3203726, -6.4037957, 16.6205502, -18.9778023, 11.7241688

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061557, upper bound: 14.7504249
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8054537, upper bound: 14.7504249
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -158.0958862, 101.5190430, -61.1898651, 24.4817467, -182.5776367, 162.7089081
1: -10.6306725, 8.0169411, -3.0392938, 3.1128674, -13.7435398, 11.0562315
2: -6.5397549, 13.3341599, -2.3409739, 3.6340690, -10.1738243, 15.6751337
3: -8.5218105, 21.6891632, -2.6836782, 7.2653589, -15.7871666, 24.3728409
4: -5.5607414, 13.8617725, -1.9518619, 4.0945668, -9.6553078, 15.8136330

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8161020
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8161428
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -158.1702118, 101.6894455, -61.1898651, 24.4817467, -182.6519623, 162.8793030
1: -10.6415453, 8.0193691, -3.0392938, 3.1128674, -13.7544127, 11.0586624
2: -6.5527611, 13.3468990, -2.3409739, 3.6340690, -10.1868305, 15.6878729
3: -8.5527124, 21.7059326, -2.6836782, 7.2653589, -15.8180695, 24.3896103
4: -5.5600491, 13.8811426, -1.9518619, 4.0945668, -9.6546154, 15.8330030

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8071237
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8075303
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -169.7490692, 115.9615860, -77.1478119, 34.7145233, -204.4635925, 193.1093903
1: -11.9737225, 8.6390867, -4.1263742, 3.8999236, -15.8736458, 12.7654600
2: -7.0982075, 14.9567432, -3.0006537, 5.1204267, -12.2186337, 17.9573975
3: -9.3590727, 23.7080593, -3.5615425, 9.5225945, -18.8816681, 27.2696018
4: -6.0562954, 15.5357952, -2.4845543, 5.6786580, -11.7349529, 18.0203457

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185417
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185417
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -169.8638611, 116.0087204, -77.1478119, 34.7145233, -204.5783844, 193.1565247
1: -11.9732084, 8.6436005, -4.1263742, 3.8999236, -15.8731318, 12.7699747
2: -7.1082392, 14.9549751, -3.0006537, 5.1204267, -12.2286654, 17.9556293
3: -9.3819017, 23.7195320, -3.5615425, 9.5225945, -18.9044952, 27.2810745
4: -6.0526004, 15.5402708, -2.4845543, 5.6786580, -11.7312584, 18.0248222

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185040
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185040
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -157.2798309, 103.7962112, -61.1898651, 24.4817467, -181.7615814, 164.9860840
1: -10.8038492, 7.9749355, -3.0392938, 3.1128674, -13.9167166, 11.0142288
2: -6.5555820, 13.6393032, -2.3409739, 3.6340690, -10.1896515, 15.9802771
3: -8.6861916, 21.7801628, -2.6836782, 7.2653589, -15.9515495, 24.4638405
4: -5.6268845, 14.1642990, -1.9518619, 4.0945668, -9.7214508, 16.1161613

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8092887
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8095564
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -154.9644165, 101.7135925, -61.1898651, 24.4817467, -179.4461670, 162.9034424
1: -10.6206255, 7.8564620, -3.0392938, 3.1128674, -13.7334929, 10.8957558
2: -6.4690018, 13.4137344, -2.3409739, 3.6340690, -10.1030712, 15.7547064
3: -8.5697670, 21.4348907, -2.6836782, 7.2653589, -15.8351250, 24.1185684
4: -5.5436792, 13.9370880, -1.9518619, 4.0945668, -9.6382446, 15.8889503

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8072805
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8073309
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -167.9698334, 117.7438049, -77.1478119, 34.7145233, -202.6843567, 194.8916168
1: -12.1114426, 8.5448055, -4.1263742, 3.8999236, -16.0113659, 12.6711788
2: -7.0867739, 15.2129297, -3.0006537, 5.1204267, -12.2072010, 18.2135830
3: -9.4919090, 23.6920128, -3.5615425, 9.5225945, -19.0145035, 27.2535553
4: -6.0985622, 15.7889242, -2.4845543, 5.6786580, -11.7772198, 18.2734699

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8127826
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8127826
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -166.2313538, 115.8847961, -77.1478119, 34.7145233, -200.9458771, 193.0325928
1: -11.9406395, 8.4592562, -4.1263742, 3.8999236, -15.8405609, 12.5856304
2: -7.0099707, 15.0049763, -3.0006537, 5.1204267, -12.1303959, 18.0056305
3: -9.3856878, 23.4091835, -3.5615425, 9.5225945, -18.9082832, 26.9707260
4: -6.0258675, 15.5815811, -2.4845543, 5.6786580, -11.7045250, 18.0661354

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8121202
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8121202
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -149.2253265, 87.1455231, -178.9485779, 126.1504974, -275.3758240, 266.0941162
1: -9.4412823, 7.5558248, -12.9268885, 9.1260834, -18.5673656, 20.4827137
2: -6.0624752, 11.8235073, -7.5344663, 16.1233349, -22.1858101, 19.3579731
3: -7.7272968, 19.9222450, -10.0063629, 25.2291126, -32.9564095, 29.9286079
4: -5.1470461, 12.3866520, -6.4307871, 16.7431602, -21.8902054, 18.8174400

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8012039
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8012039
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -170.7326202, 115.7337494, -178.9485779, 126.1504974, -296.8831177, 294.6823120
1: -11.9696894, 8.6844711, -12.9268885, 9.1260834, -21.0957718, 21.6113586
2: -7.1201820, 14.9507437, -7.5344663, 16.1233349, -23.2435150, 22.4852085
3: -9.3688164, 23.7795200, -10.0063629, 25.2291126, -34.5979309, 33.7858810
4: -6.0768752, 15.5221128, -6.4307871, 16.7431602, -22.8200359, 21.9528999

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8142219
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8142219
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -149.2253265, 87.1455231, -175.9706421, 126.8217239, -276.0470276, 263.1161499
1: -9.4412823, 7.5558248, -12.9638186, 8.9737997, -18.4150810, 20.5196438
2: -6.0624752, 11.8235073, -7.4703002, 16.2491837, -22.3116589, 19.2938061
3: -7.7272968, 19.9222450, -10.0635452, 25.0271416, -32.7544403, 29.9857883
4: -5.1470461, 12.3866520, -6.4292040, 16.8677425, -22.0147896, 18.8158531

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -170.7326202, 115.7337494, -175.9706421, 126.8217239, -297.5543518, 291.7044067
1: -11.9696894, 8.6844711, -12.9638186, 8.9737997, -20.9434853, 21.6482887
2: -7.1201820, 14.9507437, -7.4703002, 16.2491837, -23.3693657, 22.4210377
3: -9.3688164, 23.7795200, -10.0635452, 25.0271416, -34.3959579, 33.8430634
4: -6.0768752, 15.5221128, -6.4292040, 16.8677425, -22.9446144, 21.9513130

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7624463, upper bound: 14.8134851
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8063746, upper bound: 14.8134851
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -167.9698334, 117.7438049, -178.9485779, 126.1504974, -294.1203308, 296.6923828
1: -12.1114426, 8.5448055, -12.9268885, 9.1260834, -21.2375259, 21.4716949
2: -7.0867739, 15.2129297, -7.5344663, 16.1233349, -23.2101097, 22.7473965
3: -9.4919090, 23.6920128, -10.0063629, 25.2291126, -34.7210197, 33.6983757
4: -6.0985622, 15.7889242, -6.4307871, 16.7431602, -22.8417225, 22.2197075

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7977397, upper bound: 14.8063157
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8024180, upper bound: 14.8063157
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -166.2313538, 115.8847961, -178.9485779, 126.1504974, -292.3818359, 294.8333740
1: -11.9406395, 8.4592562, -12.9268885, 9.1260834, -21.0667229, 21.3861446
2: -7.0099707, 15.0049763, -7.5344663, 16.1233349, -23.1333027, 22.5394421
3: -9.3856878, 23.4091835, -10.0063629, 25.2291126, -34.6147995, 33.4155464
4: -6.0258675, 15.5815811, -6.4307871, 16.7431602, -22.7690277, 22.0123672

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7977397, upper bound: 14.8063746
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8134851, upper bound: 14.8063746
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -167.9698334, 117.7438049, -175.9706421, 126.8217239, -294.7915344, 293.7144470
1: -12.1114426, 8.5448055, -12.9638186, 8.9737997, -21.0852413, 21.5086250
2: -7.0867739, 15.2129297, -7.4703002, 16.2491837, -23.3359566, 22.6832256
3: -9.4919090, 23.6920128, -10.0635452, 25.0271416, -34.5190506, 33.7555580
4: -6.0985622, 15.7889242, -6.4292040, 16.8677425, -22.9663029, 22.2181225

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7688262, upper bound: 14.8127474
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7688262, upper bound: 14.8127474
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -166.2313538, 115.8847961, -175.9706421, 126.8217239, -293.0530701, 291.8554382
1: -11.9406395, 8.4592562, -12.9638186, 8.9737997, -20.9144382, 21.4230747
2: -7.0099707, 15.0049763, -7.4703002, 16.2491837, -23.2591515, 22.4752750
3: -9.3856878, 23.4091835, -10.0635452, 25.0271416, -34.4128304, 33.4727287
4: -6.0258675, 15.5815811, -6.4292040, 16.8677425, -22.8936081, 22.0107841

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127474
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127474
time: 0.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.42 seconds
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7665546
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7931223
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7555339, upper bound: 14.7661633
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7555339, upper bound: 14.7931223
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7930910
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7042002, upper bound: 14.8107280
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7042002, upper bound: 14.7942578
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7555339, upper bound: 14.8107281
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7871675
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7984160, upper bound: 14.7925923
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8118611, upper bound: 14.7188680
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8007433, upper bound: 14.7188680
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7950703, upper bound: 14.7557006
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7950704, upper bound: 14.7557006
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7834507, upper bound: 14.7363767
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7834508, upper bound: 14.7363767
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8061557, upper bound: 14.7504249
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8054537, upper bound: 14.7504249
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8161020
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8161428
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8071237
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8075303
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185417
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185417
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185040
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8185040
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8092887
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8095564
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8072805
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7250711, upper bound: 14.8073309
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8127826
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8127826
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8121202
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7574373, upper bound: 14.8121202
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8012039
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8012039
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8142219
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8012039, upper bound: 14.8142219
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7624463, upper bound: 14.8134851
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8063746, upper bound: 14.8134851
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7977397, upper bound: 14.8063157
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8024180, upper bound: 14.8063157
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7977397, upper bound: 14.8063746
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8134851, upper bound: 14.8063746
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7688262, upper bound: 14.8127474
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.7688262, upper bound: 14.8127474
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127474
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 4, lower bound: -14.8127474, upper bound: 14.8127474

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -72.0984116, 29.8062210, -72.0984116, 29.8062210, -101.9046326, 101.9046326
1: -3.6660242, 3.6619985, -3.6660242, 3.6619985, -7.3280230, 7.3280230
2: -2.7508221, 4.4260740, -2.7508221, 4.4260740, -7.1768947, 7.1768956
3: -3.1673064, 8.6350651, -3.1673064, 8.6350651, -11.8023710, 11.8023701
4: -2.2876673, 4.9355497, -2.2876673, 4.9355497, -7.2232170, 7.2232170

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.6984561, upper bound: 14.7825375
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6984561, upper bound: 14.7895448
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -72.0984116, 29.8062210, -73.7722931, 32.2643433, -104.3627548, 103.5785141
1: -3.6660242, 3.6619985, -3.8820581, 3.7356277, -7.4016519, 7.5440569
2: -2.7508221, 4.4260740, -2.8422780, 4.7919307, -7.5427513, 7.2683520
3: -3.1673064, 8.6350651, -3.3325341, 9.0222855, -12.1895924, 11.9675989
4: -2.2876673, 4.9355497, -2.3572531, 5.3203726, -7.6080399, 7.2928028

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7386677, upper bound: 14.7895045
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7131404, upper bound: 14.7825375
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7131404, upper bound: 14.7895448
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -60.0679550, 23.4982872, -97.2705688, 92.3322983
1: -3.8820581, 3.7356277, -2.9304743, 3.0692520, -6.9513102, 6.6661019
2: -2.8422780, 4.7919307, -2.2942970, 3.4602544, -6.3025322, 7.0862274
3: -3.3325341, 9.0222855, -2.5974555, 7.0941954, -10.4267292, 11.6197414
4: -2.3572531, 5.3203726, -1.9219065, 3.9019690, -6.2592211, 7.2422786

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6984561, upper bound: 14.7853868
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.6984561, upper bound: 14.7799263
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -72.0984116, 29.8062210, -103.5785141, 104.3627548
1: -3.8820581, 3.7356277, -3.6660242, 3.6619985, -7.5440569, 7.4016519
2: -2.8422780, 4.7919307, -2.7508221, 4.4260740, -7.2683520, 7.5427518
3: -3.3325341, 9.0222855, -3.1673064, 8.6350651, -11.9675980, 12.1895924
4: -2.3572531, 5.3203726, -2.2876673, 4.9355497, -7.2928028, 7.6080394

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6984561, upper bound: 14.8047824
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6984561, upper bound: 14.8061378
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -59.3079529, 23.4528027, -97.2250977, 91.5722961
1: -3.8820581, 3.7356277, -2.9284151, 3.0196221, -6.9016800, 6.6640425
2: -2.8422780, 4.7919307, -2.2548537, 3.4889143, -6.3311920, 7.0467834
3: -3.3325341, 9.0222855, -2.5633979, 7.0197477, -10.3522797, 11.5856838
4: -2.3572531, 5.3203726, -1.8874598, 3.9327717, -6.2900238, 7.2078319

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7131404, upper bound: 14.7848572
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7131404, upper bound: 14.7793331
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -73.7722931, 32.2643433, -106.0366287, 106.0366287
1: -3.8820581, 3.7356277, -3.8820581, 3.7356277, -7.6176853, 7.6176858
2: -2.8422780, 4.7919307, -2.8422780, 4.7919307, -7.6342082, 7.6342082
3: -3.3325341, 9.0222855, -3.3325341, 9.0222855, -12.3548193, 12.3548193
4: -2.3572531, 5.3203726, -2.3572531, 5.3203726, -7.6776257, 7.6776257

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7131404, upper bound: 14.8047824
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7131404, upper bound: 14.8061378
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -71.3583679, 28.1626167, -147.9130554, 85.4529648, -156.8113098, 176.0756683
1: -3.5577404, 3.6309237, -9.2661400, 7.4757829, -11.0335236, 12.8970633
2: -2.6802974, 4.2135391, -5.9579992, 11.6258726, -14.3061695, 10.1715364
3: -3.0570858, 8.4137774, -7.5636072, 19.6393681, -22.6964531, 15.9773846
4: -2.2340636, 4.7260466, -5.0738549, 12.1642942, -14.3983574, 9.7999010

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -71.3583679, 28.1626167, -164.5808716, 108.1314926, -179.4898376, 192.7434845
1: -3.5577404, 3.6309237, -11.2364054, 8.3580732, -11.9158134, 14.8673286
2: -2.6802974, 4.2135391, -6.7964144, 14.2230473, -16.9033451, 11.0099516
3: -3.0570858, 8.4137774, -8.9499798, 22.7231560, -25.7802391, 17.3637562
4: -2.2340636, 4.7260466, -5.8434525, 14.7446156, -16.9786797, 10.5694981

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -85.8713150, 35.4057808, -147.9130554, 85.4529648, -171.3242798, 183.3188324
1: -4.3858232, 4.3748875, -9.2661400, 7.4757829, -11.8616066, 13.6410255
2: -3.2892232, 5.2422342, -5.9579992, 11.6258726, -14.9150963, 11.2002306
3: -3.8124676, 10.2811842, -7.5636072, 19.6393681, -23.4518356, 17.8447914
4: -2.7362194, 5.8653440, -5.0738549, 12.1642942, -14.9005136, 10.9391966

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -85.8713150, 35.4057808, -164.5808716, 108.1314926, -194.0028076, 199.9866486
1: -4.3858232, 4.3748875, -11.2364054, 8.3580732, -12.7438965, 15.6112928
2: -3.2892232, 5.2422342, -6.7964144, 14.2230473, -17.5122700, 12.0386438
3: -3.8124676, 10.2811842, -8.9499798, 22.7231560, -26.5356236, 19.2311554
4: -2.7362194, 5.8653440, -5.8434525, 14.7446156, -17.4808350, 11.7087936

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -71.3583679, 28.1626167, -168.7308655, 100.5654831, -171.9238434, 196.8934784
1: -3.5577404, 3.6309237, -10.7459469, 8.5523653, -12.1101055, 14.3768711
2: -2.6802974, 4.2135391, -6.8519945, 13.4426432, -16.1229401, 11.0655336
3: -3.0570858, 8.4137774, -8.7807083, 22.5672455, -25.6243286, 17.1944847
4: -2.2340636, 4.7260466, -5.8193617, 14.0010328, -16.2350941, 10.5454063

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -71.3583679, 28.1626167, -187.5160370, 125.1257858, -196.4841614, 215.6786499
1: -3.5577404, 3.6309237, -12.8904867, 9.5454903, -13.1032305, 16.5214100
2: -2.6802974, 4.2135391, -7.7826376, 16.2574902, -18.9377880, 11.9961767
3: -3.0570858, 8.4137774, -10.2546091, 25.9713345, -29.0284195, 18.6683826
4: -2.2340636, 4.7260466, -6.6636443, 16.8426762, -19.0767403, 11.3896894

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -85.8517609, 35.3826256, -168.7308655, 100.5654831, -186.4172363, 204.1134949
1: -4.3837962, 4.3737221, -10.7459469, 8.5523653, -12.9361610, 15.1196690
2: -3.2880149, 5.2395525, -6.8519945, 13.4426432, -16.7306576, 12.0915451
3: -3.8089497, 10.2776937, -8.7807083, 22.5672455, -26.3761940, 19.0584030
4: -2.7346220, 5.8627987, -5.8193617, 14.0010328, -16.7356548, 11.6821594

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -85.8517609, 35.3826256, -187.5160370, 125.1257858, -210.9775391, 222.8986664
1: -4.3837962, 4.3737221, -12.8904867, 9.5454903, -13.9292870, 17.2642078
2: -3.2880149, 5.2395525, -7.7826376, 16.2574902, -19.5455055, 13.0221891
3: -3.8089497, 10.2776937, -10.2546091, 25.9713345, -29.7802849, 20.5323029
4: -2.7346220, 5.8627987, -6.6636443, 16.8426762, -19.5772972, 12.5264435

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -59.3079529, 23.4528027, -157.7508087, 100.5651779, -159.8731384, 181.2036133
1: -2.9284151, 3.0196221, -10.5398226, 8.0006323, -10.9290476, 13.5594435
2: -2.2548537, 3.4889143, -6.5178270, 13.2752495, -15.5301037, 10.0067415
3: -2.5633979, 7.0197477, -8.4938040, 21.6093674, -24.1727619, 15.5135489
4: -1.8874598, 3.9327717, -5.5497665, 13.7982464, -15.6857061, 9.4825382

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118255, upper bound: 14.7188680
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118255, upper bound: 14.7188680
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -59.3079529, 23.4528027, -157.9124756, 100.8531647, -160.1611176, 181.3652649
1: -2.9284151, 3.0196221, -10.5626831, 8.0072651, -10.9356804, 13.5823030
2: -2.2548537, 3.4889143, -6.5357485, 13.2998838, -15.5547371, 10.0246620
3: -2.5633979, 7.0197477, -8.5321941, 21.6422195, -24.2056160, 15.5519400
4: -1.8874598, 3.9327717, -5.5525928, 13.8291311, -15.7165909, 9.4853649

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8003128, upper bound: 14.7188680
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8003128, upper bound: 14.7188680
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -148.8415070, 86.7667007, -160.5389709, 181.1058502
1: -3.8820581, 3.7356277, -9.4078178, 7.5373106, -11.4193687, 13.1434460
2: -2.8422780, 4.7919307, -6.0417547, 11.7781420, -14.6204176, 10.8336849
3: -3.3325341, 9.0222855, -7.6923285, 19.8571243, -23.1896591, 16.7146130
4: -2.3572531, 5.3203726, -5.1294932, 12.3409510, -14.6982031, 10.4498653

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950703, upper bound: 14.7557006
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950703, upper bound: 14.7557006
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -169.7665710, 112.7354355, -186.5077209, 202.0309143
1: -3.8820581, 3.7356277, -11.6867657, 8.6383362, -12.5203943, 15.4223938
2: -2.8422780, 4.7919307, -7.0569954, 14.7778788, -17.6201553, 11.8489265
3: -3.3325341, 9.0222855, -9.2893105, 23.5459290, -26.8784618, 18.3115959
4: -2.3572531, 5.3203726, -6.0465055, 15.3251534, -17.6824074, 11.3668785

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950703, upper bound: 14.7557006
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950703, upper bound: 14.7557006
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -166.8654785, 114.2950745, -188.0673676, 199.1298218
1: -3.8820581, 3.7356277, -11.7821627, 8.4924793, -12.3745375, 15.5177898
2: -2.8422780, 4.7919307, -7.0145550, 15.0118122, -17.8540859, 11.8064861
3: -3.3325341, 9.0222855, -9.3995886, 23.4226360, -26.7551689, 18.4218750
4: -2.3572531, 5.3203726, -6.0636072, 15.5623503, -17.9196033, 11.3839798

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061557, upper bound: 14.7504249
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061557, upper bound: 14.7504249
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -73.7722931, 32.2643433, -165.3105316, 112.7714844, -186.5437775, 197.5748749
1: -3.8820581, 3.7356277, -11.6469193, 8.4160728, -12.2981310, 15.3825474
2: -2.8422780, 4.7919307, -6.9492874, 14.8326178, -17.6748924, 11.7412186
3: -3.3325341, 9.0222855, -9.3106804, 23.1801815, -26.5127144, 18.3329659
4: -2.3572531, 5.3203726, -5.9984884, 15.3869209, -17.7441750, 11.3188610

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8003128, upper bound: 14.7188680
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8054537, upper bound: 14.7504249
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -158.0958862, 101.5190430, -56.7747192, 22.3477573, -180.4436493, 158.2937622
1: -10.6306725, 8.0169411, -2.8053360, 2.9001646, -13.5308371, 10.8222742
2: -6.5397549, 13.3341599, -2.1693568, 3.3213663, -9.8611212, 15.5035143
3: -8.5218105, 21.6891632, -2.4622934, 6.7180629, -15.2398739, 24.1514549
4: -5.5607414, 13.8617725, -1.8075011, 3.7491705, -9.3099117, 15.6692724

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7121235, upper bound: 14.8016191
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7121235, upper bound: 14.8045312
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -158.0958862, 101.5190430, -60.7619934, 23.9071255, -182.0030060, 162.2810364
1: -10.6306725, 8.0169411, -3.0127778, 3.0962143, -13.7268867, 11.0297174
2: -6.5397549, 13.3341599, -2.3163786, 3.5995259, -10.1392803, 15.6505356
3: -8.5218105, 21.6891632, -2.6642110, 7.1849985, -15.7068090, 24.3533745
4: -5.5607414, 13.8617725, -1.9249959, 4.0401115, -9.6008530, 15.7867680

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7121235, upper bound: 14.8016191
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7121235, upper bound: 14.8045312
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -158.1702118, 101.6894455, -56.7747192, 22.3477573, -180.5179749, 158.4641724
1: -10.6415453, 8.0193691, -2.8053360, 2.9001646, -13.5417099, 10.8247051
2: -6.5527611, 13.3468990, -2.1693568, 3.3213663, -9.8741274, 15.5162544
3: -8.5527124, 21.7059326, -2.4622934, 6.7180629, -15.2707748, 24.1682243
4: -5.5600491, 13.8811426, -1.8075011, 3.7491705, -9.3092194, 15.6886415

Time for backsubstitution: 3.06 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.62 + 415.76 = 420.39 seconds
