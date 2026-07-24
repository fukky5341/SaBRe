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
execution time: IAR + RelationalAnalysis = 2.99 + 1.51 = 4.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8206852

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8195538
time: 0.50 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.25 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8195538
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -134.9925842, 81.8052826, -149.4919434, 98.3400955, -233.3326263, 231.2972260
1: -8.7787800, 6.8611393, -10.2893686, 7.6158886, -16.3946686, 17.1505032
2: -5.3828287, 10.9627657, -6.0750151, 12.7892494, -18.1720772, 17.0377789
3: -6.9027848, 18.0386868, -7.9261875, 20.4676228, -27.3704071, 25.9648724
4: -4.6419497, 11.4447546, -5.2263708, 13.2979355, -17.9398842, 16.6711254

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8151103
time: 0.52 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.6870610, upper bound: 14.7739433
time: 0.50 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -142.8622437, 89.6920776, -149.4919434, 98.3400955, -241.2023010, 239.1840210
1: -9.5332241, 7.2697968, -10.2893686, 7.6158886, -17.1491127, 17.5591660
2: -5.7380843, 11.9041405, -6.0750151, 12.7892494, -18.5273342, 17.9791508
3: -7.4830828, 19.2699814, -7.9261875, 20.4676228, -27.9507046, 27.1961689
4: -4.9649253, 12.4453926, -5.2263708, 13.2979355, -18.2628613, 17.6717644

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7792585
time: 0.52 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8124015, upper bound: 14.8124015
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.06 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8151103
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 4.06
Output dim: 4, lower bound: -14.6870610, upper bound: 14.7739433
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7792585
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 4, lower bound: -14.8124015, upper bound: 14.8124015

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -134.9925842, 81.8052826, -137.0211334, 81.5423279, -216.5348663, 218.8264160
1: -8.7787800, 6.8611393, -8.7929583, 6.9581327, -15.7369127, 15.6540937
2: -5.3828287, 10.9627657, -5.4053297, 10.9437370, -16.3265648, 16.3680954
3: -6.9027848, 18.0386868, -6.9033308, 18.1477737, -25.0505581, 24.9420147
4: -4.6419497, 11.4447546, -4.6565504, 11.4327507, -16.0746994, 16.1013050

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8131467
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8144848
time: 0.50 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -138.1642609, 81.0071411, -149.4919434, 98.3400955, -236.5042725, 230.4990692
1: -8.7152138, 6.9621706, -10.2893686, 7.6158886, -16.3311005, 17.2515373
2: -5.5019484, 10.9991016, -6.0750151, 12.7892494, -18.2911987, 17.0741158
3: -7.0376682, 18.3853531, -7.9261875, 20.4676228, -27.5052910, 26.3115406
4: -4.7120290, 11.4016190, -5.2263708, 13.2979355, -18.0099621, 16.6279888

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7672797, upper bound: 14.7767402
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7792585
time: 0.54 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -139.1237488, 85.0266571, -149.4919434, 98.3400955, -237.4638062, 234.5185852
1: -9.1034660, 7.0710506, -10.2893686, 7.6158886, -16.7193546, 17.3604183
2: -5.5591340, 11.3977613, -6.0750151, 12.7892494, -18.3483829, 17.4727764
3: -7.2149038, 18.6241493, -7.9261875, 20.4676228, -27.6825256, 26.5503349
4: -4.8066473, 11.9257364, -5.2263708, 13.2979355, -18.1045837, 17.1521053

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
time: 0.50 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7606364, upper bound: 14.7606364
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.03 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8131467
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8144848
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 4.03
Output dim: 4, lower bound: -14.7672797, upper bound: 14.7767402
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7792585
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
NS_A2_A2_B2, status: Status.VERIFIED, split count: 3, time: 4.03
Output dim: 4, lower bound: -14.7606364, upper bound: 14.7606364

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -118.9875870, 63.2149124, -136.9182739, 150.6184692
1: -3.8550036, 3.7463624, -7.0763807, 6.0121961, -9.8671989, 10.8227415
2: -2.8335838, 4.7218590, -4.5866833, 8.7497663, -11.5833502, 9.3085413
3: -3.3052120, 8.9624577, -5.6976700, 15.2292347, -18.5344410, 14.6601267
4: -2.3414721, 5.2556934, -3.9041324, 9.2165804, -11.5580521, 9.1598263

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8124329
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8131467
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -167.3021545, 123.2779388, -134.2860260, 77.9758148, -245.2779694, 257.5639648
1: -12.4653320, 8.5217419, -8.4795380, 6.8120613, -19.2773933, 17.0012779
2: -7.1826701, 15.7156610, -5.2528939, 10.5426550, -17.7253189, 20.9685555
3: -9.7080822, 24.0861702, -6.6461535, 17.6369038, -27.3449860, 30.7323208
4: -6.1549397, 16.2062569, -4.5235901, 11.0169420, -17.1718788, 20.7298470

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8144848
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8144848
time: 0.49 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -136.3510437, 78.5807648, -195.6805420, 144.9261017, -281.2771606, 274.2612915
1: -8.5046444, 6.8675404, -14.6865873, 9.9890881, -18.4937248, 21.5541267
2: -5.3967204, 10.7373629, -8.3289146, 18.2123413, -23.6090603, 19.0662766
3: -6.8635292, 18.0465298, -11.1344271, 27.9955597, -34.8590851, 29.1809540
4: -4.6246781, 11.1401520, -7.0948596, 18.8732567, -23.4979324, 18.2350121

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8107558, upper bound: 14.7769601
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7755390
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -139.1237488, 85.0266571, -148.1728210, 96.5252609, -235.6490021, 233.1994476
1: -9.1034660, 7.0710506, -10.1263218, 7.5451035, -16.6485691, 17.1973705
2: -5.5591340, 11.3977613, -6.0083408, 12.5890512, -18.1481857, 17.4061012
3: -7.2149038, 18.6241493, -7.8226519, 20.2275944, -27.4424973, 26.4467983
4: -4.8066473, 11.9257364, -5.1684484, 13.0920029, -17.8986492, 17.0941830

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7595257
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
time: 0.54 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
time: 0.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.90 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8124329
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8131467
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8144848
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.8077695, upper bound: 14.8144848
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.8107558, upper bound: 14.7769601
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7755390
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.90
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -74.1615601, 30.8451347, -104.5484848, 105.7924500
1: -3.8550036, 3.7463624, -3.8129649, 3.7640061, -7.6190100, 7.5593271
2: -2.8335838, 4.7218590, -2.8003910, 4.6389937, -7.4725771, 7.5222502
3: -3.3052120, 8.9624577, -3.2553163, 8.9069939, -12.2122059, 12.2177734
4: -2.3414721, 5.2556934, -2.3287997, 5.1645155, -7.5059876, 7.5844932

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7656405, upper bound: 14.7991598
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7690781, upper bound: 14.8090300
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -153.4582214, 98.4333572, -172.1367188, 185.0891113
1: -3.8550036, 3.7463624, -10.3260508, 7.7363129, -11.5913162, 14.0724134
2: -2.8335838, 4.7218590, -6.3208585, 13.1641884, -15.9977713, 11.0427151
3: -3.3052120, 8.9624577, -8.3833475, 21.1586227, -24.4638329, 17.3458061
4: -2.3414721, 5.2556934, -5.4720945, 13.3829670, -15.7244396, 10.7277880

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8131467
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8131467
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -152.6789856, 101.9078979, -134.2860260, 77.9758148, -230.6548004, 236.1939240
1: -10.5914364, 7.7514067, -8.4795380, 6.8120613, -17.4034977, 16.2309437
2: -6.3533502, 13.4016705, -5.2528939, 10.5426550, -16.8960037, 18.6545639
3: -8.4281673, 21.2625523, -6.6461535, 17.6369038, -26.0650711, 27.9087029
4: -5.4749856, 13.8828840, -4.5235901, 11.0169420, -16.4919281, 18.4064751

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8128843
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8138344
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -170.0235138, 119.7905579, -134.2860260, 77.9758148, -247.9993286, 254.0765839
1: -12.1958609, 8.6541214, -8.4795380, 6.8120613, -19.0079212, 17.1336555
2: -7.2145205, 15.3963356, -5.2528939, 10.5426550, -17.7571716, 20.6492290
3: -9.6822062, 24.0892105, -6.6461535, 17.6369038, -27.3191109, 30.7353611
4: -6.1819458, 15.8992510, -4.5235901, 11.0169420, -17.1988869, 20.4228401

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8128843
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8138344
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -136.6060944, 82.0715332, -193.3067169, 142.4287415, -279.0348511, 275.3782349
1: -8.7700424, 6.8951797, -14.4487305, 9.8659964, -18.6360359, 21.3439102
2: -5.4588885, 11.0737705, -8.2145863, 17.9118290, -23.3707180, 19.2883549
3: -7.0000892, 18.2536564, -10.9627934, 27.5993271, -34.5994148, 29.2164478
4: -4.6904049, 11.3871708, -6.9975080, 18.5640831, -23.2544880, 18.3846779

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7900162, upper bound: 14.7668115
time: 0.50 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8084966, upper bound: 14.7668115
time: 0.52 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -135.3593140, 77.4164047, -195.6805420, 144.9261017, -280.2854004, 273.0969543
1: -8.4020309, 6.8166685, -14.6865873, 9.9890881, -18.3911152, 21.5032520
2: -5.3468881, 10.5986948, -8.3289146, 18.2123413, -23.5592251, 18.9276085
3: -6.7797241, 17.8723183, -11.1344271, 27.9955597, -34.7752838, 29.0067444
4: -4.5793238, 11.0114737, -7.0948596, 18.8732567, -23.4525795, 18.1063328

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7755390
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7755390
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -139.1237488, 85.0266571, -133.7698059, 80.0439148, -219.1676636, 218.7964630
1: -9.1034660, 7.0710506, -8.6227922, 6.7951827, -15.8986492, 15.6938429
2: -5.5591340, 11.3977613, -5.3191676, 10.7682619, -16.3273964, 16.7169285
3: -7.2149038, 18.6241493, -6.8036113, 17.8090038, -25.0239067, 25.4277534
4: -4.8066473, 11.9257364, -4.5868263, 11.2466869, -16.0533333, 16.5125599

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7992569, upper bound: 14.7606364
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -139.1237488, 85.0266571, -141.6768799, 87.9700928, -227.0938416, 226.7035217
1: -9.1034660, 7.0710506, -9.3809872, 7.2061286, -16.3095932, 16.4520378
2: -5.5591340, 11.3977613, -5.6761017, 11.7155066, -17.2746372, 17.0738602
3: -7.2149038, 18.6241493, -7.3853130, 19.0463390, -26.2612419, 26.0094547
4: -4.8066473, 11.9257364, -4.9109230, 12.2540932, -17.0607414, 16.8366566

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8036480, upper bound: 14.7603865
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865
time: 0.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.68 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7656405, upper bound: 14.7991598
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7690781, upper bound: 14.8090300
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8131467
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8131467
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8128843
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8138344
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8128843
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8138344
NS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7900162, upper bound: 14.7668115
NS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.8084966, upper bound: 14.7668115
NS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7755390
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.8121210, upper bound: 14.7755390
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.8048155, upper bound: 14.7606364
NS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.8036480, upper bound: 14.7603865
NS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.68
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -71.6873016, 29.1020393, -102.8054047, 103.3181915
1: -3.8550036, 3.7463624, -3.6409795, 3.6373019, -7.4923058, 7.3873420
2: -2.8335838, 4.7218590, -2.6958315, 4.3846474, -7.2182307, 7.4176898
3: -3.3052120, 8.9624577, -3.1008148, 8.5346260, -11.8398380, 12.0632725
4: -2.3414721, 5.2556934, -2.2366352, 4.8922195, -7.2336917, 7.4923282

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7361942, upper bound: 14.7101731
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7477860, upper bound: 14.7933421
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -67.0984039, 26.8745155, -100.5778809, 98.7292938
1: -3.8550036, 3.7463624, -3.3770821, 3.3977294, -7.2527332, 7.1234446
2: -2.8335838, 4.7218590, -2.5126681, 4.0563765, -6.8899603, 7.2345266
3: -3.3052120, 8.9624577, -2.8539152, 7.9584365, -11.2636471, 11.8163710
4: -2.3414721, 5.2556934, -2.0775051, 4.5261173, -6.8675890, 7.3331985

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7621467, upper bound: 14.7859058
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7621467, upper bound: 14.8029546
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -143.0570374, 88.0249100, -161.7282715, 174.6879272
1: -3.8550036, 3.7463624, -9.3824549, 7.2262301, -11.0812340, 13.1288176
2: -2.8335838, 4.7218590, -5.8160095, 11.9077101, -14.7412930, 10.5378685
3: -3.3052120, 8.9624577, -7.6045046, 19.4468670, -22.7520733, 16.5669613
4: -2.3414721, 5.2556934, -5.0272651, 12.2192497, -14.5607224, 10.2829590

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7702293, upper bound: 14.7679036
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8074926, upper bound: 14.8112717
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -147.9928131, 91.1982269, -164.9015808, 179.6237030
1: -3.8550036, 3.7463624, -9.7251120, 7.4700375, -11.3250408, 13.4714737
2: -2.8335838, 4.7218590, -5.9987493, 12.3723526, -15.2059364, 10.7206078
3: -3.3052120, 8.9624577, -7.9247723, 20.0684032, -23.3736115, 16.8872299
4: -2.3414721, 5.2556934, -5.2199187, 12.6495352, -14.9910069, 10.4756117

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7702293, upper bound: 14.7688501
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8074926, upper bound: 14.8112717
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -152.6789856, 101.9078979, -74.1615601, 30.8451347, -183.5241241, 176.0694275
1: -10.5914364, 7.7514067, -3.8129649, 3.7640061, -14.3554411, 11.5643711
2: -6.3533502, 13.4016705, -2.8003910, 4.6389937, -10.9923439, 16.2020607
3: -8.4281673, 21.2625523, -3.2553163, 8.9069939, -17.3351612, 24.5178680
4: -5.4749856, 13.8828840, -2.3287997, 5.1645155, -10.6395006, 16.2116833

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7942908
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8091651
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -152.6789856, 101.9078979, -155.8390808, 105.2988663, -257.9778137, 257.7469482
1: -10.5914364, 7.7514067, -10.8338833, 7.9173236, -18.5087605, 18.5852871
2: -6.3533502, 13.4016705, -6.4757905, 13.8434649, -20.1968155, 19.8774548
3: -8.4281673, 21.2625523, -8.5942612, 21.7346287, -30.1627960, 29.8568134
4: -5.4749856, 13.8828840, -5.5982924, 14.3119860, -19.7869720, 19.4811764

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8149591
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8149591
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -170.0235138, 119.7905579, -74.1615601, 30.8451347, -200.8686371, 193.9520569
1: -12.1958609, 8.6541214, -3.8129649, 3.7640061, -15.9598665, 12.4670868
2: -7.2145205, 15.3963356, -2.8003910, 4.6389937, -11.8535137, 18.1967258
3: -9.6822062, 24.0892105, -3.2553163, 8.9069939, -18.5891991, 27.3445263
4: -6.1819458, 15.8992510, -2.3287997, 5.1645155, -11.3464603, 18.2280502

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7941100
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8067466
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -170.0235138, 119.7905579, -155.8390808, 105.2988663, -275.3223877, 275.6295776
1: -12.1958609, 8.6541214, -10.8338833, 7.9173236, -20.1131840, 19.4880028
2: -7.2145205, 15.3963356, -6.4757905, 13.8434649, -21.0579853, 21.8721256
3: -9.6822062, 24.0892105, -8.5942612, 21.7346287, -31.4168358, 32.6834717
4: -6.1819458, 15.8992510, -5.5982924, 14.3119860, -20.4939308, 21.4975433

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7969487
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8076350
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -127.7459869, 71.5920258, -135.7660217, 74.4552536, -202.2012177, 207.3580475
1: -7.8137217, 6.4390802, -8.1042891, 6.8576698, -14.6713915, 14.5433683
2: -5.0201650, 9.7882233, -5.4801984, 10.0179510, -15.0381165, 15.2684202
3: -6.2754803, 16.7022743, -6.7550583, 17.6877174, -23.9631977, 23.4573307
4: -4.2741723, 10.2096205, -4.5352077, 10.7897062, -15.0638781, 14.7448273

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7720923, upper bound: 14.7668115
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7720923, upper bound: 14.7668115
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -136.6060944, 82.0715332, -186.5639191, 135.1888733, -271.7949829, 268.6354370
1: -8.7700424, 6.8951797, -13.7620230, 9.5138912, -18.2839317, 20.6572018
2: -5.4588885, 11.0737705, -7.8949208, 17.0916862, -22.5505753, 18.9686871
3: -7.0000892, 18.2536564, -10.5051279, 26.5111656, -33.5112534, 28.7587814
4: -4.6904049, 11.3871708, -6.7278771, 17.7167015, -22.4071064, 18.1150475

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8074395, upper bound: 14.7668115
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7869545, upper bound: 14.7638421
time: 0.51 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7668075, upper bound: 14.7668115
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7668075, upper bound: 14.7668115
time: 0.48 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -135.3593140, 77.4164047, -179.4290466, 126.6083527, -261.9676514, 256.8454590
1: -8.4020309, 6.8166685, -12.9703083, 9.1483183, -17.5503483, 19.7869759
2: -5.3468881, 10.5986948, -7.5566487, 16.1841888, -21.5310726, 18.1553402
3: -6.7797241, 17.8723183, -10.0429697, 25.3084297, -32.0881538, 27.9152870
4: -4.5793238, 11.0114737, -6.4515276, 16.8023033, -21.3816223, 17.4630013

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8114493, upper bound: 14.7714207
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7752001
time: 0.55 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7755390
time: 0.62 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -135.3593140, 77.4164047, -176.4409790, 127.2970657, -262.6563721, 253.8573608
1: -8.4020309, 6.8166685, -13.0086794, 8.9960995, -17.3981304, 19.8253460
2: -5.3468881, 10.5986948, -7.4931507, 16.3116360, -21.6585217, 18.0918465
3: -6.7797241, 17.8723183, -10.1010818, 25.1094704, -31.8891907, 27.9734001
4: -4.5793238, 11.0114737, -6.4500360, 16.9278450, -21.5071678, 17.4615097

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8114493, upper bound: 14.7725618
time: 0.51 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7752001
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7755390
time: 0.55 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -137.9523468, 83.3307877, -133.7698059, 80.0439148, -217.9962616, 217.1005859
1: -8.9530220, 7.0082283, -8.6227922, 6.7951827, -15.7482052, 15.6310205
2: -5.4981899, 11.2109823, -5.3191676, 10.7682619, -16.2664528, 16.5301495
3: -7.1196566, 18.4033184, -6.8036113, 17.8090038, -24.9286575, 25.2069283
4: -4.7537718, 11.7365017, -4.5868263, 11.2466869, -16.0004578, 16.3233280

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073397, upper bound: 14.7606364
time: 0.53 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8022707, upper bound: 14.7606364
time: 0.54 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -140.2645416, 82.9056091, -133.7698059, 80.0439148, -220.3084564, 216.6754150
1: -8.9456196, 7.1028419, -8.6227922, 6.7951827, -15.7408028, 15.7256336
2: -5.6423445, 11.2300167, -5.3191676, 10.7682619, -16.4106064, 16.5491848
3: -7.3960171, 18.6766605, -6.8036113, 17.8090038, -25.2050190, 25.4802666
4: -4.8697782, 11.7562084, -4.5868263, 11.2466869, -16.1164646, 16.3430328

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8055241, upper bound: 14.7603082
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -139.1237488, 85.0266571, -139.0246277, 83.8120575, -222.9358063, 224.0512543
1: -9.1034660, 7.0710506, -9.0133801, 7.0638227, -16.1672897, 16.0844307
2: -5.5591340, 11.3977613, -5.5331106, 11.2605515, -16.8196831, 16.9308720
3: -7.2149038, 18.6241493, -7.1496978, 18.5194778, -25.7343826, 25.7738400
4: -4.8066473, 11.9257364, -4.7837629, 11.7973728, -16.6040154, 16.7094975

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8036480, upper bound: 14.7603865
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8036480, upper bound: 14.7603865
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -139.1237488, 85.0266571, -134.3495483, 77.6068115, -216.7305298, 219.3761444
1: -9.1034660, 7.0710506, -8.4630718, 6.8104444, -15.9139080, 15.5341225
2: -5.5591340, 11.3977613, -5.2914824, 10.5239086, -16.0830402, 16.6892433
3: -7.2149038, 18.6241493, -6.7332602, 17.6515083, -24.8664112, 25.3574066
4: -4.8066473, 11.9257364, -4.5578499, 11.0483541, -15.8550014, 16.4835854

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865
time: 0.50 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865
time: 0.55 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.74 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7361942, upper bound: 14.7101731
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7477860, upper bound: 14.7933421
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7621467, upper bound: 14.7859058
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7621467, upper bound: 14.8029546
NS_A1_B1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7702293, upper bound: 14.7679036
NS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.8074926, upper bound: 14.8112717
NS_A1_B1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7702293, upper bound: 14.7688501
NS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.8074926, upper bound: 14.8112717
NS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7942908
NS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8091651
NS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8149591
NS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7742162, upper bound: 14.8149591
NS_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7941100
NS_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8067466
NS_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7969487
NS_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8076350
NS_A2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7720923, upper bound: 14.7668115
NS_A2_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7720923, upper bound: 14.7668115
NS_A2_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7668075, upper bound: 14.7668115
NS_A2_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7668075, upper bound: 14.7668115
NS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7752001
NS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7755390
NS_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7752001
NS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7755390
NS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.8073397, upper bound: 14.7606364
NS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.8022707, upper bound: 14.7606364
NS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.8036480, upper bound: 14.7603865
NS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.8036480, upper bound: 14.7603865
NS_A2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865
NS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.74
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -71.2067108, 30.3328857, -71.6873016, 29.1020393, -100.3087463, 102.0201874
1: -3.7220001, 3.6210721, -3.6409795, 3.6373019, -7.3593020, 7.2620511
2: -2.7274463, 4.5687842, -2.6958315, 4.3846474, -7.1120934, 7.2646155
3: -3.1853838, 8.6383057, -3.1008148, 8.5346260, -11.7200098, 11.7391205
4: -2.2642565, 5.0808268, -2.2366352, 4.8922195, -7.1564760, 7.3174615

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7364795, upper bound: 14.7676196
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7364795, upper bound: 14.7933423
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -68.1762695, 26.8774834, -67.0984039, 26.8745155, -95.0507812, 93.9758911
1: -3.4006052, 3.4836540, -3.3770821, 3.3977294, -6.7983341, 6.8607354
2: -2.5815849, 4.0262613, -2.5126681, 4.0563765, -6.6379614, 6.5389295
3: -2.9215953, 8.0517864, -2.8539152, 7.9584365, -10.8800316, 10.9056988
4: -2.1449318, 4.5174136, -2.0775051, 4.5261173, -6.6710491, 6.5949187

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7621467, upper bound: 14.7859058
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7621467, upper bound: 14.7859058
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -70.4991226, 29.3105812, -67.0984039, 26.8745155, -97.3736191, 96.4089737
1: -3.6235805, 3.5893874, -3.3770821, 3.3977294, -7.0213094, 6.9664679
2: -2.6832647, 4.4123173, -2.5126681, 4.0563765, -6.7396412, 6.9249849
3: -3.0933795, 8.4799900, -2.8539152, 7.9584365, -11.0518150, 11.3339052
4: -2.2254598, 4.9056892, -2.0775051, 4.5261173, -6.7515764, 6.9831944

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7621467, upper bound: 14.8029546
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7621467, upper bound: 14.8029546
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -139.0402527, 84.2896500, -157.9929962, 170.6711426
1: -3.8550036, 3.7463624, -9.0301342, 7.0223875, -10.8773909, 12.7764969
2: -2.8335838, 4.7218590, -5.6263967, 11.4589968, -14.2925806, 10.3482533
3: -3.3052120, 8.9624577, -7.3229542, 18.8119030, -22.1171112, 16.2854080
4: -2.3414721, 5.2556934, -4.8619552, 11.7998753, -14.1413479, 10.1176472

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138674, upper bound: 14.8077637
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8143173, upper bound: 14.8077637
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -73.7033615, 31.6308899, -143.1076202, 86.7182693, -160.4216309, 174.7385101
1: -3.8550036, 3.7463624, -9.3228655, 7.2311983, -11.0862007, 13.0692263
2: -2.8335838, 4.7218590, -5.7748022, 11.8516512, -14.6852350, 10.4966612
3: -3.3052120, 8.9624577, -7.5930262, 19.3048859, -22.6100941, 16.5554829
4: -2.3414721, 5.2556934, -5.0253668, 12.1637325, -14.5052042, 10.2810602

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7670309, upper bound: 14.8077637
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8066080, upper bound: 14.8077637
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -133.0374451, 71.4529343, -74.1615601, 30.8451347, -163.8825836, 145.6144714
1: -7.9988627, 6.7355943, -3.8129649, 3.7640061, -11.7628679, 10.5485592
2: -5.2757745, 9.9500589, -2.8003910, 4.6389937, -9.9147673, 12.7504501
3: -6.5913701, 17.2923470, -3.2553163, 8.9069939, -15.4983635, 20.5476627
4: -4.4906602, 10.5958338, -2.3287997, 5.1645155, -9.6551752, 12.9246330

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7942908
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7942908
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -145.5421600, 92.0948334, -74.1615601, 30.8451347, -176.3872986, 166.2563782
1: -9.7044210, 7.3699951, -3.8129649, 3.7640061, -13.4684267, 11.1829596
2: -5.9787273, 12.2981882, -2.8003910, 4.6389937, -10.6177197, 15.0985794
3: -7.8344045, 19.9369926, -3.2553163, 8.9069939, -16.7413979, 23.1923084
4: -5.1449661, 12.7442217, -2.3287997, 5.1645155, -10.3094788, 15.0730209

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7523844, upper bound: 14.7851229
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7523844, upper bound: 14.8091651
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -152.6789856, 101.9078979, -144.2544556, 91.6415100, -244.3204956, 246.1623535
1: -10.5914364, 7.7514067, -9.6467142, 7.3173265, -17.9087639, 17.3981209
2: -6.3533502, 13.4016705, -5.9006844, 12.2638569, -18.6172066, 19.3023510
3: -8.4281673, 21.2625523, -7.7081056, 19.7509441, -28.1791115, 28.9706554
4: -5.4749856, 13.8828840, -5.0957212, 12.7285461, -18.2035313, 18.9786034

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8032917, upper bound: 14.7870989
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8074926, upper bound: 14.8149591
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -152.6789856, 101.9078979, -149.5906067, 96.1615829, -248.8405762, 251.4984894
1: -10.5914364, 7.7514067, -10.0871487, 7.5933671, -18.1848030, 17.8385544
2: -6.3533502, 13.4016705, -6.1137180, 12.8634291, -19.2167797, 19.5153847
3: -8.4281673, 21.2625523, -8.0644207, 20.4861050, -28.9142723, 29.3269730
4: -5.4749856, 13.8828840, -5.3130503, 13.3689470, -18.8439331, 19.1959343

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8032917, upper bound: 14.7870989
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8074926, upper bound: 14.8149591
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -149.6574249, 85.8249893, -74.1615601, 30.8451347, -180.5025635, 159.9865112
1: -9.3017139, 7.5840235, -3.8129649, 3.7640061, -13.0657187, 11.3969879
2: -6.0643873, 11.6512642, -2.8003910, 4.6389937, -10.7033806, 14.4516544
3: -7.7174015, 19.8415508, -3.2553163, 8.9069939, -16.6243954, 23.0968666
4: -5.1449375, 12.2654161, -2.3287997, 5.1645155, -10.3094501, 14.5942154

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7941100
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.7941100
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -162.3290558, 108.6713333, -74.1615601, 30.8451347, -193.1741943, 182.8328705
1: -11.1967583, 8.2414408, -3.8129649, 3.7640061, -14.9607639, 12.0544033
2: -6.7934065, 14.1510296, -2.8003910, 4.6389937, -11.4323997, 16.9514198
3: -9.0099392, 22.6147518, -3.2553163, 8.9069939, -17.9169331, 25.8700676
4: -5.8204694, 14.6183319, -2.3287997, 5.1645155, -10.9849825, 16.9471302

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8067466
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8067466
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -149.6574249, 85.8249893, -155.8390808, 105.2988663, -254.9562531, 241.6640625
1: -9.3017139, 7.5840235, -10.8338833, 7.9173236, -17.2190361, 18.4179058
2: -6.0643873, 11.6512642, -6.4757905, 13.8434649, -19.9078522, 18.1270504
3: -7.7174015, 19.8415508, -8.5942612, 21.7346287, -29.4520302, 28.4358120
4: -5.1449375, 12.2654161, -5.5982924, 14.3119860, -19.4569244, 17.8637085

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7795937, upper bound: 14.7969487
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7795937, upper bound: 14.7969487
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -162.3290558, 108.6713333, -155.8390808, 105.2988663, -267.6279297, 264.5104065
1: -11.1967583, 8.2414408, -10.8338833, 7.9173236, -19.1140823, 19.0753250
2: -6.7934065, 14.1510296, -6.4757905, 13.8434649, -20.6368713, 20.6268196
3: -9.0099392, 22.6147518, -8.5942612, 21.7346287, -30.7445679, 31.2090130
4: -5.8204694, 14.6183319, -5.5982924, 14.3119860, -20.1324558, 20.2166252

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7673046, upper bound: 14.8076351
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7795917, upper bound: 14.8076350
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -134.4629669, 76.4289627, -179.4290466, 126.6083527, -261.0713196, 255.8580017
1: -8.3142824, 6.7711000, -12.9703083, 9.1483183, -17.4625988, 19.7414093
2: -5.3013821, 10.4811945, -7.5566487, 16.1841888, -21.4855709, 18.0378437
3: -6.7066913, 17.7179737, -10.0429697, 25.3084297, -32.0151215, 27.7609444
4: -4.5389242, 10.8985367, -6.4515276, 16.8023033, -21.3412247, 17.3500633

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7901420, upper bound: 14.7727788
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7881415, upper bound: 14.7727788
time: 0.54 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -136.1228485, 79.5787888, -179.4290466, 126.6083527, -262.7312012, 259.0078125
1: -8.5718565, 6.8576193, -12.9703083, 9.1483183, -17.7201710, 19.8279266
2: -5.3938313, 10.8317118, -7.5566487, 16.1841888, -21.5780201, 18.3883572
3: -6.8719101, 18.0848808, -10.0429697, 25.3084297, -32.1803360, 28.1278496
4: -4.6278758, 11.1890850, -6.4515276, 16.8023033, -21.4301720, 17.6406136

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8001849, upper bound: 14.7730844
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8018529, upper bound: 14.7755390
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8018529, upper bound: 14.7755390
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -134.4629669, 76.4289627, -176.4409790, 127.2970657, -261.7600403, 252.8699341
1: -8.3142824, 6.7711000, -13.0086794, 8.9960995, -17.3103809, 19.7797794
2: -5.3013821, 10.4811945, -7.4931507, 16.3116360, -21.6130180, 17.9743462
3: -6.7066913, 17.7179737, -10.1010818, 25.1094704, -31.8161583, 27.8190556
4: -4.5389242, 10.8985367, -6.4500360, 16.9278450, -21.4667702, 17.3485718

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7895289, upper bound: 14.7727788
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7950443, upper bound: 14.7727306
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7752001
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7752001
time: 0.54 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -136.1228485, 79.5787888, -176.4409790, 127.2970657, -263.4199219, 256.0197449
1: -8.5718565, 6.8576193, -13.0086794, 8.9960995, -17.5679550, 19.8662968
2: -5.3938313, 10.8317118, -7.4931507, 16.3116360, -21.7054672, 18.3248577
3: -6.8719101, 18.0848808, -10.1010818, 25.1094704, -31.9813766, 28.1859627
4: -4.6278758, 11.1890850, -6.4500360, 16.9278450, -21.5557194, 17.6391220

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7755390
time: 0.55 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7993917, upper bound: 14.7755390
time: 0.51 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -137.9523468, 83.3307877, -132.5854034, 78.4821167, -216.4344635, 215.9161835
1: -8.9530220, 7.0082283, -8.4847174, 6.7326035, -15.6856251, 15.4929457
2: -5.4981899, 11.2109823, -5.2562218, 10.5928764, -16.0910645, 16.4672050
3: -7.1196566, 18.4033184, -6.7050939, 17.5907974, -24.7104530, 25.1084118
4: -4.7537718, 11.7365017, -4.5324297, 11.0691099, -15.8228817, 16.2689323

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8025280, upper bound: 14.7996103
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8029782, upper bound: 14.7996103
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -137.9523468, 83.3307877, -134.4580536, 82.8276138, -220.7799683, 217.7888489
1: -8.9530220, 7.0082283, -8.8526773, 6.8393173, -15.7923393, 15.8609056
2: -5.4981899, 11.2109823, -5.3692117, 11.0356407, -16.5338287, 16.5801945
3: -7.1196566, 18.4033184, -6.8871832, 18.0340271, -25.1536827, 25.2905006
4: -4.7537718, 11.7365017, -4.6311369, 11.5140247, -16.2677956, 16.3676376

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8029782, upper bound: 14.7996103
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8029782, upper bound: 14.7996103
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -137.9523468, 83.3307877, -139.0246277, 83.8120575, -221.7644043, 222.3553925
1: -8.9530220, 7.0082283, -9.0133801, 7.0638227, -16.0168457, 16.0216084
2: -5.4981899, 11.2109823, -5.5331106, 11.2605515, -16.7587395, 16.7440929
3: -7.1196566, 18.4033184, -7.1496978, 18.5194778, -25.6391335, 25.5530167
4: -4.7537718, 11.7365017, -4.7837629, 11.7973728, -16.5511436, 16.5202637

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8019770, upper bound: 14.7600818
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -140.2645416, 82.9056091, -139.0246277, 83.8120575, -224.0765991, 221.9302216
1: -8.9456196, 7.1028419, -9.0133801, 7.0638227, -16.0094414, 16.1162205
2: -5.6423445, 11.2300167, -5.5331106, 11.2605515, -16.9028931, 16.7631264
3: -7.3960171, 18.6766605, -7.1496978, 18.5194778, -25.9154949, 25.8263550
4: -4.8697782, 11.7562084, -4.7837629, 11.7973728, -16.6671505, 16.5399704

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8019770, upper bound: 14.7600818
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -137.9523468, 83.3307877, -134.3495483, 77.6068115, -215.5591583, 217.6802979
1: -8.9530220, 7.0082283, -8.4630718, 6.8104444, -15.7634659, 15.4713001
2: -5.4981899, 11.2109823, -5.2914824, 10.5239086, -16.0220985, 16.5024643
3: -7.1196566, 18.4033184, -6.7332602, 17.6515083, -24.7711639, 25.1365776
4: -4.7537718, 11.7365017, -4.5578499, 11.0483541, -15.8021259, 16.2943497

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7922105, upper bound: 14.7603865
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7943498, upper bound: 14.7603865
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -140.2645416, 82.9056091, -134.3495483, 77.6068115, -217.8713531, 217.2551117
1: -8.9456196, 7.1028419, -8.4630718, 6.8104444, -15.7560616, 15.5659132
2: -5.6423445, 11.2300167, -5.2914824, 10.5239086, -16.1662521, 16.5214996
3: -7.3960171, 18.6766605, -6.7332602, 17.6515083, -25.0475254, 25.4099197
4: -4.8697782, 11.7562084, -4.5578499, 11.0483541, -15.9181328, 16.3140564

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7918916, upper bound: 14.7600818
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.50 + 305.77 = 310.27 seconds
