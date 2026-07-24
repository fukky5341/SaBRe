## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 19178.25882359392


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9290.7187500, 11623.0966797, -9290.7187500, 11623.0966797, -20913.8105469, 20913.8105469)
1: (-1088.3400879, 983.5498047, -1088.3400879, 983.5498047, -2071.8898926, 2071.8898926)
2: (-636.5715942, 1120.6903076, -636.5715942, 1120.6903076, -1757.2619629, 1757.2619629)
3: (-516.6026001, 1142.5196533, -516.6026001, 1142.5196533, -1659.1223145, 1659.1223145)
4: (-748.2526855, 957.9287109, -748.2526855, 957.9287109, -1706.1810303, 1706.1810303)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.59 + 2.08 = 4.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -19178.4506081, upper bound: 19178.4506081

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4481650
time: 0.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4486794
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4481650
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4486794

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8501.3271484, 10635.9677734, -8912.4843750, 11110.8701172, -19612.1933594, 19548.4511719
1: -997.4971313, 899.4035034, -1040.7940674, 941.2086182, -1938.7056885, 1940.1973877
2: -581.7885742, 1027.5623779, -609.1723022, 1073.0576172, -1654.8459473, 1636.7346191
3: -472.2976990, 1045.2779541, -494.9886475, 1092.3656006, -1564.6632080, 1540.2666016
4: -684.1896362, 877.9801025, -716.0427856, 917.2218628, -1601.4113770, 1594.0227051

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4484914, upper bound: 19178.4471326
time: 0.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481419, upper bound: 19178.4471326
time: 0.64 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9218.1230469, 11543.0771484, -9252.5019531, 11581.0781250, -20799.1972656, 20795.5761719
1: -1081.0371094, 976.3294678, -1084.5161133, 979.7409668, -2060.7780762, 2060.8454590
2: -631.9311523, 1112.7808838, -634.1304932, 1116.5422363, -1748.4733887, 1746.9112549
3: -512.6972046, 1134.5822754, -514.5364380, 1138.3519287, -1651.0490723, 1649.1185303
4: -742.8044434, 951.0460815, -745.3854980, 954.3115234, -1697.1157227, 1696.4316406

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4481799
time: 0.71 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799
time: 1.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.46 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 0, lower bound: -19178.4484914, upper bound: 19178.4471326
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 0, lower bound: -19178.4481419, upper bound: 19178.4471326
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 0, lower bound: -19178.4486794, upper bound: 19178.4481799
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 0, lower bound: -19178.4481799, upper bound: 19178.4481799

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8361.4804688, 10460.5830078, -8684.8681641, 10816.3896484, -19177.8691406, 19145.4511719
1: -981.0081787, 884.4586792, -1013.1224365, 916.3854370, -1897.3935547, 1897.5810547
2: -572.1428833, 1010.6484985, -593.2042236, 1044.7564697, -1616.8990479, 1603.8526611
3: -464.4511414, 1028.2684326, -482.1402283, 1063.8955078, -1528.3465576, 1510.4085693
4: -672.8152466, 863.3761597, -697.1749878, 892.8503418, -1565.6655273, 1560.5511475

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4465384, upper bound: 19178.4463292
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481419, upper bound: 19178.4471326
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481419, upper bound: 19178.4471326
time: 0.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8372.7900391, 10513.5175781, -9261.2890625, 11268.6660156, -19641.4570312, 19774.8046875
1: -986.1185303, 887.7056274, -1051.8769531, 969.3424072, -1955.4609375, 1939.5825195
2: -574.0765991, 1014.7514648, -626.2780151, 1093.5191650, -1667.5957031, 1641.0294189
3: -465.5458679, 1032.7830811, -516.4442139, 1110.1895752, -1575.7352295, 1549.2272949
4: -675.1770630, 866.9627075, -736.4785767, 938.4669189, -1613.6440430, 1603.4412842

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4442925, upper bound: 19178.4467965
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4478320, upper bound: 19178.4471326
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4478320, upper bound: 19178.4468157
time: 0.65 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8955.7587891, 11195.7900391, -8854.0546875, 11057.6552734, -20013.4121094, 20049.8437500
1: -1048.0310059, 947.4883423, -1034.7840576, 936.1520996, -1984.1831055, 1982.2724609
2: -613.4862671, 1079.2053223, -606.2349854, 1065.7900391, -1679.2762451, 1685.4401855
3: -498.0445862, 1100.4040527, -492.3110657, 1086.8013916, -1584.8459473, 1592.7149658
4: -720.7917480, 922.5816040, -712.1006470, 911.2557373, -1632.0474854, 1634.6822510

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4481120
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480881, upper bound: 19178.4476929
time: 0.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8950.1308594, 11223.7412109, -8991.4023438, 11234.5166016, -20184.6484375, 20215.1445312
1: -1051.2727051, 948.5356445, -1051.7282715, 950.9704590, -2002.2431641, 2000.2639160
2: -614.0065308, 1081.3961182, -615.4457397, 1083.0744629, -1697.0808105, 1696.8417969
3: -497.8834229, 1102.9880371, -500.0823975, 1104.2485352, -1602.1315918, 1603.0701904
4: -721.6341553, 924.1210938, -723.5762329, 925.7434082, -1647.3775635, 1647.6972656

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481799
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4476929
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.05 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4481419, upper bound: 19178.4471326
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4481419, upper bound: 19178.4471326
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4478320, upper bound: 19178.4471326
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4478320, upper bound: 19178.4468157
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4481120
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4480881, upper bound: 19178.4476929
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481799
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4476929

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8261.1347656, 10334.3603516, -8684.8681641, 10816.3896484, -19077.5234375, 19019.2285156
1: -969.2061157, 873.7606812, -1013.1224365, 916.3854370, -1885.5915527, 1886.8830566
2: -565.2129517, 998.4318237, -593.2042236, 1044.7564697, -1609.9692383, 1591.6357422
3: -458.8227539, 1016.0312500, -482.1402283, 1063.8955078, -1522.7181396, 1498.1712646
4: -664.6198120, 852.8638916, -697.1749878, 892.8503418, -1557.4696045, 1550.0388184

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8898.0253906, 10824.7890625, -8684.8681641, 10816.3896484, -19714.4140625, 19509.6562500
1: -1011.1287231, 931.2961426, -1013.1224365, 916.3854370, -1927.5141602, 1944.4185791
2: -601.6301270, 1051.3770752, -593.2042236, 1044.7564697, -1646.3863525, 1644.5811768
3: -496.3642883, 1066.1411133, -482.1402283, 1063.8955078, -1560.2597656, 1548.2813721
4: -707.9166260, 902.6045532, -697.1749878, 892.8503418, -1600.7668457, 1599.7795410

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8135.4252930, 10223.9912109, -9149.0566406, 11125.9599609, -19261.3847656, 19373.0429688
1: -958.7836304, 862.7447510, -1038.2874756, 957.3731689, -1916.1564941, 1901.0321045
2: -557.9777832, 986.4194946, -618.5042725, 1079.6102295, -1637.5880127, 1604.9237061
3: -452.3912964, 1004.3580322, -510.2286682, 1096.2416992, -1548.6330566, 1514.5866699
4: -656.3355103, 842.4811401, -727.4638062, 926.5787354, -1582.9141846, 1569.9447021

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4471326
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4471326
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8952.5283203, 11089.1650391, -9200.1640625, 11211.3173828, -20163.8457031, 20289.3281250
1: -1039.6413574, 942.3173218, -1046.5705566, 963.7611694, -2003.4022217, 1988.8879395
2: -608.8861084, 1074.1453857, -622.6489868, 1087.5594482, -1696.4455566, 1696.7944336
3: -497.0713806, 1091.1254883, -513.2389526, 1104.2994385, -1601.3708496, 1604.3640137
4: -716.3392334, 917.7653198, -732.2741089, 933.2470093, -1649.5860596, 1650.0391846

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4468157
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4468157
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8718.8896484, 10894.2500000, -8715.1376953, 10879.8085938, -19598.6992188, 19609.3867188
1: -1019.6737671, 921.8909912, -1017.9999390, 921.0968628, -1940.7706299, 1939.8908691
2: -597.0265503, 1050.2033691, -596.5576172, 1048.7144775, -1645.7409668, 1646.7607422
3: -484.5625610, 1071.2022705, -484.4063721, 1069.5751953, -1554.1376953, 1555.6085205
4: -701.3320312, 897.5667114, -700.6668091, 896.5310669, -1597.8630371, 1598.2335205

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4460837, upper bound: 19178.4460682
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4476929
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4476929
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9201.3339844, 11265.0830078, -8723.0195312, 10931.8876953, -20133.2226562, 19988.1015625
1: -1052.2597656, 965.4221191, -1023.1923828, 924.1380615, -1976.3978271, 1988.6142578
2: -624.4771118, 1091.0076904, -598.3451538, 1052.7327881, -1677.2099609, 1689.3525391
3: -512.6406860, 1109.4851074, -485.3979492, 1074.0914307, -1586.7321777, 1594.8828125
4: -733.3608398, 935.7513428, -702.8648071, 899.9641724, -1633.3248291, 1638.6158447

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4473497, upper bound: 19178.4423569
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8717.4160156, 10927.5996094, -8854.7558594, 11058.8564453, -19776.2734375, 19782.3496094
1: -1023.3765869, 923.3937378, -1035.1387939, 936.1340942, -1959.5106201, 1958.5324707
2: -597.8393555, 1052.9295654, -605.9075928, 1066.2138672, -1664.0532227, 1658.8367920
3: -484.6376953, 1074.3074951, -492.4002380, 1087.2453613, -1571.8830566, 1566.7076416
4: -702.5408325, 899.5643921, -712.3187866, 911.2138062, -1613.7541504, 1611.8830566

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4475897, upper bound: 19178.4481799
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4475552, upper bound: 19178.4481234
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4476929
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4476929
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9127.1806641, 11212.6943359, -8849.3564453, 11094.7089844, -20221.8886719, 20062.0507812
1: -1047.5512695, 959.4770508, -1038.8737793, 937.7799072, -1985.3311768, 1998.3508301
2: -620.4329224, 1085.0532227, -606.7950439, 1068.6956787, -1689.1284180, 1691.8482666
3: -508.5387878, 1104.2735596, -492.4563293, 1090.2104492, -1598.7492676, 1596.7296143
4: -728.8840332, 930.0191040, -713.3893433, 913.3355103, -1642.2193604, 1643.4084473

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4470480, upper bound: 19178.4423569
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
time: 1.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.30 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4480457, upper bound: 19178.4471326
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4471326
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4471326
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4468157
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4468157, upper bound: 19178.4468157
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4476929
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4476929
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4473497, upper bound: 19178.4423569
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4476929
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4476929
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4470480, upper bound: 19178.4423569
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.30
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8261.1347656, 10334.3603516, -8261.1347656, 10334.3603516, -18595.4941406, 18595.4941406
1: -969.2061157, 873.7606812, -969.2061157, 873.7606812, -1842.9667969, 1842.9667969
2: -565.2129517, 998.4318237, -565.2129517, 998.4318237, -1563.6444092, 1563.6444092
3: -458.8227539, 1016.0312500, -458.8227539, 1016.0312500, -1474.8538818, 1474.8538818
4: -664.6198120, 852.8638916, -664.6198120, 852.8638916, -1517.4835205, 1517.4835205

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4429454
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4390680
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8261.1347656, 10334.3603516, -8992.8154297, 11254.7050781, -19515.8398438, 19327.1757812
1: -969.2061157, 873.7606812, -1053.9204102, 951.8993530, -1921.1054688, 1927.6810303
2: -565.2129517, 998.4318237, -616.2164917, 1085.0524902, -1650.2652588, 1614.6479492
3: -458.8227539, 1016.0312500, -499.8552246, 1106.6770020, -1565.4997559, 1515.8863525
4: -664.6198120, 852.8638916, -724.2387695, 927.1442261, -1591.7637939, 1577.1026611

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4429454
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4390680
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8898.0253906, 10824.7890625, -8261.1347656, 10334.3603516, -19232.3867188, 19085.9238281
1: -1011.1287231, 931.2961426, -969.2061157, 873.7606812, -1884.8894043, 1900.5021973
2: -601.6301270, 1051.3770752, -565.2129517, 998.4318237, -1600.0615234, 1616.5897217
3: -496.3642883, 1066.1411133, -458.8227539, 1016.0312500, -1512.3955078, 1524.9638672
4: -707.9166260, 902.6045532, -664.6198120, 852.8638916, -1560.7805176, 1567.2238770

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8898.0253906, 10824.7890625, -8992.8154297, 11254.7050781, -20152.7304688, 19817.6054688
1: -1011.1287231, 931.2961426, -1053.9204102, 951.8993530, -1963.0280762, 1985.2164307
2: -601.6301270, 1051.3770752, -616.2164917, 1085.0524902, -1686.6823730, 1667.5932617
3: -496.3642883, 1066.1411133, -499.8552246, 1106.6770020, -1603.0412598, 1565.9963379
4: -707.9166260, 902.6045532, -724.2387695, 927.1442261, -1635.0607910, 1626.8432617

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8135.4252930, 10223.9912109, -8787.9707031, 10684.0253906, -18819.4511719, 19011.9609375
1: -958.7836304, 862.7447510, -997.6842041, 919.5361938, -1878.3194580, 1860.4289551
2: -557.9777832, 986.4194946, -594.0427856, 1037.6713867, -1595.6491699, 1580.4622803
3: -452.3912964, 1004.3580322, -490.2407227, 1052.4588623, -1504.8500977, 1494.5987549
4: -656.3355103, 842.4811401, -699.0331421, 890.9331055, -1547.2685547, 1541.5142822

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4467579
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4454094
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8135.4252930, 10223.9912109, -9304.3857422, 11411.2265625, -19546.6503906, 19528.3769531
1: -958.7836304, 862.7447510, -1066.1236572, 977.2412109, -1936.0245361, 1928.8684082
2: -557.9777832, 986.4194946, -631.9185181, 1104.9746094, -1662.9522705, 1618.3380127
3: -452.3912964, 1004.3580322, -518.3751221, 1124.0220947, -1576.4133301, 1522.7331543
4: -656.3355103, 842.4811401, -742.4915161, 947.3142090, -1603.6496582, 1584.9726562

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4467579
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4454094
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8952.5283203, 11089.1650391, -8844.4931641, 10773.6787109, -19726.2011719, 19933.6582031
1: -1039.6413574, 942.3173218, -1006.3720703, 926.3935547, -1966.0347900, 1948.6893311
2: -608.8861084, 1074.1453857, -598.4414673, 1046.1040039, -1654.9901123, 1672.5869141
3: -497.0713806, 1091.1254883, -493.5462036, 1060.9379883, -1558.0093994, 1584.6715088
4: -716.3392334, 917.7653198, -704.2178955, 898.0113525, -1614.3504639, 1621.9830322

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4428113
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4451805
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8952.5283203, 11089.1650391, -9358.7841797, 11499.8027344, -20452.3281250, 20447.9414062
1: -1039.6413574, 942.3173218, -1074.6258545, 983.9534912, -2023.5947266, 2016.9431152
2: -608.8861084, 1074.1453857, -636.2684326, 1113.2033691, -1722.0894775, 1710.4138184
3: -497.0713806, 1091.1254883, -521.5680542, 1132.3902588, -1629.4616699, 1612.6934814
4: -716.3392334, 917.7653198, -747.6171265, 954.3018188, -1670.6406250, 1665.3823242

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4428113
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4451805
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8718.8896484, 10894.2500000, -8613.6289062, 10750.8603516, -19469.7500000, 19507.8769531
1: -1019.6737671, 921.8909912, -1005.9617310, 910.1701050, -1929.8438721, 1927.8527832
2: -597.0265503, 1050.2033691, -589.5191040, 1036.2663574, -1633.2927246, 1639.7224121
3: -484.5625610, 1071.2022705, -478.6281738, 1057.0777588, -1541.6401367, 1549.8303223
4: -701.3320312, 897.5667114, -692.3291016, 885.8114624, -1587.1431885, 1589.8957520

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480839, upper bound: 19178.4481120
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4477614
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8718.8896484, 10894.2500000, -9131.2402344, 11152.8896484, -19871.7773438, 20025.4843750
1: -1019.6737671, 921.8909912, -1041.0926514, 957.1546021, -1976.8283691, 1962.9836426
2: -597.0265503, 1050.2033691, -619.0335693, 1080.3229980, -1677.3496094, 1669.2364502
3: -484.5625610, 1071.2022705, -509.0184631, 1098.4422607, -1583.0047607, 1580.2204590
4: -701.3320312, 897.5667114, -726.9398804, 926.9931641, -1628.3249512, 1624.5065918

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4481120
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4481120
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9108.9316406, 11133.9843750, -8443.4121094, 10536.1435547, -19645.0703125, 19577.3925781
1: -1039.3726807, 955.0177002, -985.4577637, 892.2416992, -1931.6143799, 1940.4754639
2: -617.7362671, 1078.4028320, -577.9141235, 1015.2765503, -1633.0128174, 1656.3168945
3: -507.4451904, 1096.7529297, -469.5285034, 1035.1983643, -1542.6435547, 1566.2814941
4: -725.4757080, 924.9773560, -678.7591553, 868.1332397, -1593.6088867, 1603.7360840

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9024.5292969, 11075.1835938, -9748.7275391, 12017.3242188, -21041.8476562, 20823.9101562
1: -1034.9792480, 948.2293091, -1123.8898926, 1023.0333252, -2058.0126953, 2072.1191406
2: -613.0277100, 1072.2512207, -661.7728271, 1162.1123047, -1775.1400146, 1734.0240479
3: -503.0113525, 1090.6765137, -541.2542725, 1183.3038330, -1686.3151855, 1631.9307861
4: -720.0115967, 919.6713867, -778.4793701, 993.2277222, -1713.2392578, 1698.1507568

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8717.4160156, 10927.5996094, -8749.1816406, 10922.7148438, -19640.1250000, 19676.7773438
1: -1023.3765869, 923.3937378, -1022.4063721, 924.6672363, -1948.0438232, 1945.8000488
2: -597.8393555, 1052.9295654, -598.5305786, 1053.1405029, -1650.9798584, 1651.4600830
3: -484.6376953, 1074.3074951, -486.4586487, 1074.0764160, -1558.7141113, 1560.7659912
4: -702.5408325, 899.5643921, -703.5883789, 899.9399414, -1602.4807129, 1603.1528320

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481799
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481120
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8717.4160156, 10927.5996094, -9165.6279297, 11221.8466797, -19939.2597656, 20093.2265625
1: -1023.3765869, 923.3937378, -1047.8450928, 962.1190796, -1985.4956055, 1971.2387695
2: -597.8393555, 1052.9295654, -621.5610352, 1086.7484131, -1684.5877686, 1674.4902344
3: -484.6376953, 1074.3074951, -511.0831909, 1105.5457764, -1590.1833496, 1585.3906250
4: -702.5408325, 899.5643921, -730.9599609, 931.3255005, -1633.8662109, 1630.5244141

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481799
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481120
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9043.9785156, 11095.4023438, -8582.4003906, 10710.6728516, -19754.6523438, 19677.8027344
1: -1036.2879639, 950.0947876, -1002.2061157, 907.0322876, -1943.3203125, 1952.3009033
2: -614.3594360, 1073.7495117, -587.1208496, 1032.4619141, -1646.8212891, 1660.8703613
3: -503.8742676, 1092.8012695, -477.3782959, 1052.5063477, -1556.3806152, 1570.1795654
4: -721.7512207, 920.3994751, -690.2055054, 882.5538940, -1604.3049316, 1610.6048584

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8934.6113281, 10987.2763672, -9804.0693359, 12058.4541016, -20993.0644531, 20791.3437500
1: -1027.3333740, 939.7004395, -1127.9282227, 1027.5092773, -2054.8425293, 2067.6284180
2: -607.5385132, 1063.5887451, -664.3850098, 1167.5268555, -1775.0654297, 1727.9737549
3: -497.9739075, 1081.8973389, -544.4281006, 1187.8911133, -1685.8648682, 1626.3254395
4: -713.6707153, 911.8212891, -782.1808472, 997.3709106, -1711.0416260, 1694.0019531

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.48 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4429454
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4390680
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4429454
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4388620, upper bound: 19178.4390680
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4467579
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4454094
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4467579
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4457377, upper bound: 19178.4454094
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4428113
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4451805
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4428113
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4449882, upper bound: 19178.4451805
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4480839, upper bound: 19178.4481120
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4477614
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4481120
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4481120
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4437234, upper bound: 19178.4423569
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481799
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481120
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481799
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4476929, upper bound: 19178.4481120
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.48
Output dim: 0, lower bound: -19178.4423569, upper bound: 19178.4423569

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8054.6049805, 10113.1093750, -8154.8828125, 10220.5302734, -18275.1347656, 18267.9882812
1: -948.6935425, 853.6156006, -958.6635132, 863.3726807, -1812.0661621, 1812.2790527
2: -552.1775513, 976.2653809, -558.5003662, 987.0322876, -1539.2098389, 1534.7657471
3: -447.5529785, 993.9088135, -453.0158386, 1004.6227417, -1452.1756592, 1446.9244385
4: -649.3560791, 833.7413330, -656.7621460, 843.0196533, -1492.3756104, 1490.5034180

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4295101, upper bound: 19178.4434194
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4100332, upper bound: 19178.4402562
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9244.4013672, 11772.0410156, -8041.0883789, 10095.3232422, -19339.7226562, 19813.1269531
1: -1106.2293701, 988.0495605, -947.1990967, 852.1715698, -1958.4008789, 1935.2486572
2: -638.1111450, 1132.9252930, -551.1824951, 974.6823120, -1612.7934570, 1684.1075439
3: -514.7376709, 1154.3054199, -446.8069763, 992.2909546, -1507.0285645, 1601.1123047
4: -751.1101685, 967.0064087, -648.2412720, 832.3226929, -1583.4328613, 1615.2476807

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4390680, upper bound: 19178.4389628
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4389628, upper bound: 19178.4389628
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8054.6049805, 10113.1093750, -8889.4775391, 11143.6943359, -19198.2949219, 19002.5859375
1: -948.6935425, 853.6156006, -1043.6281738, 941.8612061, -1890.5546875, 1897.2437744
2: -552.1775513, 976.2653809, -609.6823730, 1073.9296875, -1626.1070557, 1585.9477539
3: -447.5529785, 993.9088135, -494.2642822, 1095.6033936, -1543.1562500, 1488.1730957
4: -649.3560791, 833.7413330, -716.6146240, 917.5470581, -1566.9029541, 1550.3559570

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4394102, upper bound: 19178.4409920
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4397904, upper bound: 19178.4390680
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4397904, upper bound: 19178.4390680
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9244.4013672, 11772.0410156, -8761.2353516, 11001.8701172, -20246.2714844, 20533.2753906
1: -1106.2293701, 988.0495605, -1030.6791992, 929.1171265, -2035.3464355, 2018.7286377
2: -638.1111450, 1132.9252930, -601.4209595, 1059.9637451, -1698.0749512, 1734.3461914
3: -514.7376709, 1154.3054199, -487.1570129, 1081.6590576, -1596.3967285, 1641.4622803
4: -751.1101685, 967.0064087, -706.8436279, 905.4496460, -1656.5598145, 1673.8498535

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4394001, upper bound: 19178.4389628
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4397904, upper bound: 19178.4390680
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4397904, upper bound: 19178.4390680
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8102.0786133, 10173.8935547, -8787.9707031, 10684.0253906, -18786.1035156, 18961.8613281
1: -953.9912109, 858.8198853, -997.6842041, 919.5361938, -1873.5273438, 1856.5041504
2: -555.4768677, 981.6743774, -594.0427856, 1037.6713867, -1593.1481934, 1575.7171631
3: -450.4891052, 999.4947510, -490.2407227, 1052.4588623, -1502.9479980, 1489.7354736
4: -653.3355713, 838.4933472, -699.0331421, 890.9331055, -1544.2685547, 1537.5263672

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4454094, upper bound: 19178.4467579
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4454094, upper bound: 19178.4467579
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8177.0971680, 10227.6328125, -8740.1162109, 10615.7070312, -18792.8027344, 18967.7500000
1: -958.6386719, 864.9022827, -991.3047485, 914.0648193, -1872.7033691, 1856.2070312
2: -559.3699951, 987.7369385, -590.5939941, 1031.2521973, -1590.6221924, 1578.3308105
3: -454.4550476, 1005.2359009, -487.5681152, 1045.8695068, -1500.3245850, 1492.8038330
4: -657.9318237, 843.8010864, -694.8671265, 885.6053467, -1543.5371094, 1538.6680908

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4454094, upper bound: 19178.4454094
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4454094, upper bound: 19178.4454094
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8102.0786133, 10173.8935547, -9304.3857422, 11411.2265625, -19513.3046875, 19478.2753906
1: -953.9912109, 858.8198853, -1066.1236572, 977.2412109, -1931.2324219, 1924.9436035
2: -555.4768677, 981.6743774, -631.9185181, 1104.9746094, -1660.4514160, 1613.5928955
3: -450.4891052, 999.4947510, -518.3751221, 1124.0220947, -1574.5112305, 1517.8698730
4: -653.3355713, 838.4933472, -742.4915161, 947.3142090, -1600.6497803, 1580.9848633

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4466224, upper bound: 19178.4454094
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4466224, upper bound: 19178.4454094
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8177.0971680, 10227.6328125, -9254.7441406, 11338.9033203, -19516.0000000, 19482.3769531
1: -958.6386719, 864.9022827, -1059.4703369, 971.5236816, -1930.1622314, 1924.3724365
2: -559.3699951, 987.7369385, -628.2781982, 1098.2181396, -1657.5881348, 1616.0151367
3: -454.4550476, 1005.2359009, -515.5912476, 1116.9648438, -1571.4199219, 1520.8270264
4: -657.9318237, 843.8010864, -738.1080933, 941.7583008, -1599.6901855, 1581.9091797

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4466224, upper bound: 19178.4454094
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4466224, upper bound: 19178.4454094
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8363.8271484, 10428.3496094, -8539.1308594, 10450.1767578, -18814.0039062, 18967.4804688
1: -978.5260010, 882.8635864, -977.2910767, 896.3244629, -1874.8504639, 1860.1546631
2: -570.7963257, 1008.3340454, -578.9674072, 1013.8558350, -1584.6520996, 1587.3012695
3: -464.1570740, 1025.4658203, -476.4564819, 1028.6094971, -1492.7666016, 1501.9223633
4: -671.4194946, 860.8371582, -681.1902466, 869.8637085, -1541.2832031, 1542.0273438

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4451805, upper bound: 19178.4428113
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4451805, upper bound: 19178.4428113
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8862.2480469, 10993.9492188, -8801.5195312, 10726.5488281, -19588.7910156, 19795.4609375
1: -1031.0174561, 933.6369019, -1002.0033569, 922.2059326, -1953.2233887, 1935.6402588
2: -603.0419312, 1064.6776123, -595.6718750, 1041.3880615, -1644.4299316, 1660.3494873
3: -492.2317505, 1081.6740723, -491.2449036, 1056.2817383, -1548.5130615, 1572.9188232
4: -709.6450195, 909.4354248, -701.0178223, 893.9671631, -1603.6120605, 1610.4526367

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4451805, upper bound: 19178.4451805
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4451805, upper bound: 19178.4451805
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8363.8271484, 10428.3496094, -9074.3193359, 11190.5976562, -19554.4257812, 19502.6679688
1: -978.5260010, 882.8635864, -1046.5961914, 955.6679688, -1934.1939697, 1929.4597168
2: -570.7963257, 1008.3340454, -618.1248169, 1082.4006348, -1653.1970215, 1626.4588623
3: -464.1570740, 1025.4658203, -505.6430969, 1101.5848389, -1565.7419434, 1531.1088867
4: -671.4194946, 860.8371582, -726.1122437, 927.7391357, -1599.1586914, 1586.9494629

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449938, upper bound: 19178.4428113
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449938, upper bound: 19178.4428113
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8862.2480469, 10993.9492188, -9306.3808594, 11445.2021484, -20307.4492188, 20300.3242188
1: -1031.0174561, 933.6369019, -1069.6354980, 978.9425659, -2009.9599609, 2003.2724609
2: -603.0419312, 1064.6776123, -632.9177246, 1107.6835938, -1710.7255859, 1697.5953369
3: -492.2317505, 1081.6740723, -518.7871094, 1126.9020996, -1619.1337891, 1600.4610596
4: -709.6450195, 909.4354248, -743.7713013, 949.5090332, -1659.1536865, 1653.2062988

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449938, upper bound: 19178.4451805
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4449938, upper bound: 19178.4451805
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8409.1875000, 10499.0419922, -8420.0498047, 10505.0625000, -18914.2480469, 18919.0898438
1: -983.2203979, 888.3205566, -983.2597046, 889.2003784, -1872.4207764, 1871.5803223
2: -575.4219360, 1012.6932373, -576.0494995, 1012.9114990, -1588.3334961, 1588.7426758
3: -467.0026550, 1032.2203369, -467.6356506, 1032.8367920, -1499.8389893, 1499.8557129
4: -675.6537476, 865.5212402, -676.3267822, 865.8294067, -1541.4831543, 1541.8480225

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4485116, upper bound: 19178.4477614
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4485116, upper bound: 19178.4477614
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8656.9814453, 10824.2695312, -8488.6425781, 10618.3974609, -19275.3789062, 19312.9121094
1: -1013.5531006, 915.4156494, -993.7401733, 897.9694214, -1911.5224609, 1909.1557617
2: -592.8184814, 1043.2246094, -581.6171875, 1023.0051880, -1615.8236084, 1624.8416748
3: -481.0652771, 1064.1086426, -471.8659363, 1043.7838135, -1524.8491211, 1535.9744873
4: -696.2593994, 891.4790039, -683.0109863, 874.3067627, -1570.5660400, 1574.4898682

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4485116, upper bound: 19178.4477614
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4485116, upper bound: 19178.4477614
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8574.7500000, 10707.4443359, -9131.2402344, 11152.8896484, -19727.6347656, 19838.6816406
1: -1001.9511108, 906.2656250, -1041.0926514, 957.1546021, -1959.1057129, 1947.3582764
2: -587.0133667, 1032.0004883, -619.0335693, 1080.3229980, -1667.3364258, 1651.0335693
3: -476.5346069, 1052.7735596, -509.0184631, 1098.4422607, -1574.9768066, 1561.7918701
4: -689.3911133, 882.1060791, -726.9398804, 926.9931641, -1616.3842773, 1609.0458984

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4479840
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4479982, upper bound: 19178.4481120
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8705.0039062, 10876.5253906, -9131.2402344, 11152.8896484, -19857.8925781, 20007.7656250
1: -1018.1928711, 920.3302612, -1041.0926514, 957.1546021, -1975.3474121, 1961.4228516
2: -595.7593384, 1048.5245361, -619.0335693, 1080.3229980, -1676.0822754, 1667.5576172
3: -483.8772888, 1069.4880371, -509.0184631, 1098.4422607, -1582.3194580, 1578.5063477
4: -700.3057251, 895.9153442, -726.9398804, 926.9931641, -1627.2987061, 1622.8551025

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4480879, upper bound: 19178.4479840
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4479982, upper bound: 19178.4481120
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8978.3183594, 10950.6347656, -8443.4121094, 10536.1435547, -19514.4609375, 19394.0429688
1: -1021.7708130, 940.3837891, -985.4577637, 892.2416992, -1914.0124512, 1925.8415527
2: -608.2451782, 1060.6862793, -577.9141235, 1015.2765503, -1623.5217285, 1638.6002197
3: -500.1284790, 1078.8670654, -469.5285034, 1035.1983643, -1535.3269043, 1548.3953857
4: -714.3623047, 909.9190674, -678.7591553, 868.1332397, -1582.4956055, 1588.6778564

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4473421, upper bound: 19178.4420337
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4473281, upper bound: 19178.4420337
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10508.1552734, 12672.2724609, -8443.4121094, 10536.1435547, -21044.2988281, 21115.6816406
1: -1182.0435791, 1094.2259521, -985.4577637, 892.2416992, -2074.2851562, 2079.6833496
2: -706.4393921, 1232.0780029, -577.9141235, 1015.2765503, -1721.7159424, 1809.9920654
3: -584.8552246, 1250.6457520, -469.5285034, 1035.1983643, -1620.0535889, 1720.1741943
4: -831.2844849, 1057.0644531, -678.7591553, 868.1332397, -1699.4177246, 1735.8233643

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4473421, upper bound: 19178.4420337
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4473281, upper bound: 19178.4420337
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8893.6298828, 10899.5595703, -9748.7275391, 12017.3242188, -20910.9492188, 20648.2871094
1: -1018.1578979, 933.8023682, -1123.8898926, 1023.0333252, -2041.1911621, 2057.6918945
2: -603.7723999, 1055.2257080, -661.7728271, 1162.1123047, -1765.8847656, 1716.9985352
3: -495.7594910, 1073.3955078, -541.2542725, 1183.3038330, -1679.0633545, 1614.6495361
4: -708.9555664, 905.3217773, -778.4793701, 993.2277222, -1702.1833496, 1683.8011475

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4413581, upper bound: 19178.4414120
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4436598, upper bound: 19178.4414120
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8904.5546875, 10908.3193359, -9748.7275391, 12017.3242188, -20921.8789062, 20657.0429688
1: -1019.5091553, 934.5893555, -1123.8898926, 1023.0333252, -2042.5424805, 2058.4792480
2: -604.0245361, 1057.0909424, -661.7728271, 1162.1123047, -1766.1368408, 1718.8637695
3: -496.4127502, 1074.4898682, -541.2542725, 1183.3038330, -1679.7165527, 1615.7438965
4: -710.1837158, 906.0360107, -778.4793701, 993.2277222, -1703.4113770, 1684.5153809

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4413581, upper bound: 19178.4414120
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4436598, upper bound: 19178.4414120
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8574.7500000, 10707.4443359, -8749.1816406, 10922.7148438, -19497.4570312, 19456.6250000
1: -1001.9511108, 906.2656250, -1022.4063721, 924.6672363, -1926.6181641, 1928.6719971
2: -587.0133667, 1032.0004883, -598.5305786, 1053.1405029, -1640.1538086, 1630.5310059
3: -476.5346069, 1052.7735596, -486.4586487, 1074.0764160, -1550.6109619, 1539.2320557
4: -689.3911133, 882.1060791, -703.5883789, 899.9399414, -1589.3310547, 1585.6944580

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481293, upper bound: 19178.4477614
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4477614, upper bound: 19178.4477614
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8705.0039062, 10876.5253906, -8749.1816406, 10922.7148438, -19627.7148438, 19625.7070312
1: -1018.1928711, 920.3302612, -1022.4063721, 924.6672363, -1942.8599854, 1942.7365723
2: -595.7593384, 1048.5245361, -598.5305786, 1053.1405029, -1648.8999023, 1647.0550537
3: -483.8772888, 1069.4880371, -486.4586487, 1074.0764160, -1557.9534912, 1555.9465332
4: -700.3057251, 895.9153442, -703.5883789, 899.9399414, -1600.2456055, 1599.5036621

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4481293, upper bound: 19178.4477614
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4477614, upper bound: 19178.4477614
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8574.7500000, 10707.4443359, -9165.6279297, 11221.8466797, -19796.5937500, 19873.0722656
1: -1001.9511108, 906.2656250, -1047.8450928, 962.1190796, -1964.0701904, 1954.1107178
2: -587.0133667, 1032.0004883, -621.5610352, 1086.7484131, -1673.7617188, 1653.5612793
3: -476.5346069, 1052.7735596, -511.0831909, 1105.5457764, -1582.0802002, 1563.8566895
4: -689.3911133, 882.1060791, -730.9599609, 931.3255005, -1620.7165527, 1613.0660400

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4450076, upper bound: 19178.4455290
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4472224, upper bound: 19178.4470900
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4470657, upper bound: 19178.4470900
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8705.0039062, 10876.5253906, -9165.6279297, 11221.8466797, -19926.8515625, 20042.1523438
1: -1018.1928711, 920.3302612, -1047.8450928, 962.1190796, -1980.3120117, 1968.1752930
2: -595.7593384, 1048.5245361, -621.5610352, 1086.7484131, -1682.5078125, 1670.0853271
3: -483.8772888, 1069.4880371, -511.0831909, 1105.5457764, -1589.4228516, 1580.5712891
4: -700.3057251, 895.9153442, -730.9599609, 931.3255005, -1631.6312256, 1626.8752441

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4450076, upper bound: 19178.4453677
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4472224, upper bound: 19178.4471792
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4470657, upper bound: 19178.4471792
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8916.6279297, 10919.0283203, -8582.4003906, 10710.6728516, -19627.3007812, 19501.4296875
1: -1019.3476562, 935.8267212, -1002.2061157, 907.0322876, -1926.3796387, 1938.0328369
2: -605.1670532, 1056.6452637, -587.1208496, 1032.4619141, -1637.6289062, 1643.7658691
3: -496.7510986, 1075.5007324, -477.3782959, 1052.5063477, -1549.2574463, 1552.8789062
4: -710.9101562, 905.8209839, -690.2055054, 882.5538940, -1593.4637451, 1596.0262451

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4438421, upper bound: 19178.4143509
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4439471, upper bound: 19178.4143509
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10457.9355469, 12661.2138672, -8582.4003906, 10710.6728516, -21168.6093750, 21243.6132812
1: -1181.2631836, 1091.2321777, -1002.2061157, 907.0322876, -2088.2954102, 2093.4382324
2: -704.3881226, 1230.0886230, -587.1208496, 1032.4619141, -1736.8499756, 1817.2094727
3: -582.2502441, 1248.7742920, -477.3782959, 1052.5063477, -1634.7565918, 1726.1524658
4: -829.1450806, 1054.6597900, -690.2055054, 882.5538940, -1711.6987305, 1744.8651123

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4438421, upper bound: 19178.4143509
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4439471, upper bound: 19178.4143509
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8893.6298828, 10899.5595703, -9804.0693359, 12058.4541016, -20952.0800781, 20703.6289062
1: -1018.1578979, 933.8023682, -1127.9282227, 1027.5092773, -2045.6671143, 2061.7302246
2: -603.7723999, 1055.2257080, -664.3850098, 1167.5268555, -1771.2993164, 1719.6107178
3: -495.7594910, 1073.3955078, -544.4281006, 1187.8911133, -1683.6505127, 1617.8236084
4: -708.9555664, 905.3217773, -782.1808472, 997.3709106, -1706.3264160, 1687.5025635

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4359284, upper bound: 19178.4393934
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -19178.4393934, upper bound: 19178.4393934
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8904.5546875, 10908.3193359, -9804.0693359, 12058.4541016, -20963.0078125, 20712.3867188
1: -1019.5091553, 934.5893555, -1127.9282227, 1027.5092773, -2047.0183105, 2062.5175781
2: -604.0245361, 1057.0909424, -664.3850098, 1167.5268555, -1771.5513916, 1721.4759521
3: -496.4127502, 1074.4898682, -544.4281006, 1187.8911133, -1684.3035889, 1618.9179688
4: -710.1837158, 906.0360107, -782.1808472, 997.3709106, -1707.5546875, 1688.2167969

Time for backsubstitution: 2.64 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.68 + 417.88 = 422.56 seconds
