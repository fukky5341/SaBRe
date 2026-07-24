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
execution time: IAR + RelationalAnalysis = 1.15 + 2.28 = 3.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -42.5258574, upper bound: 42.5258574

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5252559, upper bound: 42.5253945
time: 0.82 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251771, upper bound: 42.5251771
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -42.5252559, upper bound: 42.5253945
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -42.5251771, upper bound: 42.5251771

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -24.7039509, 26.6083031, -25.2903595, 27.1793919, -51.8833427, 51.8986588
1: -193.3704834, 62.3708038, -197.2152100, 63.7769470, -257.1474304, 259.5859985
2: -103.4975052, 57.3958511, -105.7661362, 58.6568146, -162.1543274, 163.1619873
3: -134.3548889, 46.1009293, -137.1408997, 47.0920601, -181.4469147, 183.2418213
4: -72.9179077, 48.9754601, -74.5733490, 50.0467873, -122.9646912, 123.5487976

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251339, upper bound: 42.5251649
time: 0.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250031, upper bound: 42.5251529
time: 0.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -26.4428310, 28.3737679, -25.2583447, 27.0510979, -53.4939232, 53.6321068
1: -204.2971802, 66.6481934, -195.5477295, 63.7066078, -268.0037842, 262.1958923
2: -109.8801498, 61.3709793, -105.3400345, 58.5341721, -168.4143219, 166.7110138
3: -142.1901855, 49.1605530, -136.2117920, 46.9051743, -189.0953522, 185.3723450
4: -77.6663055, 52.5038338, -74.4032593, 49.9457817, -127.6120911, 126.9070816

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5250775, upper bound: 42.5249740
time: 0.90 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249412, upper bound: 42.5249412
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.15 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -42.5251339, upper bound: 42.5251649
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -42.5250031, upper bound: 42.5251529
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -42.5250775, upper bound: 42.5249740
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -42.5249412, upper bound: 42.5249412

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -24.0350952, 25.8607178, -23.7354164, 25.6127892, -49.6478806, 49.5961342
1: -188.1215210, 60.6906776, -186.1467590, 59.9468613, -248.0683746, 246.8374023
2: -100.7733459, 55.7953606, -99.1731644, 55.3935204, -156.1668701, 154.9685211
3: -130.7533722, 44.8147354, -129.2352295, 44.4545250, -175.2078857, 174.0499573
4: -70.9929276, 47.5653381, -69.9121246, 47.2826462, -118.2755737, 117.4774475

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.72 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -24.1970024, 26.0645180, -24.3600101, 26.1971035, -50.3941040, 50.4245300
1: -189.4398041, 61.0802460, -190.1724854, 61.4298897, -250.8696747, 251.2527313
2: -101.3771591, 56.1971970, -101.9239883, 56.4836082, -157.8607483, 158.1211853
3: -131.5992432, 45.1552887, -132.1719055, 45.3836899, -176.9829102, 177.3271942
4: -71.4082718, 47.9626999, -71.8157654, 48.2162819, -119.6245575, 119.7784653

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222211, upper bound: 42.5229221
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222210, upper bound: 42.5229219
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -25.7462330, 27.5942192, -23.4032288, 25.1647797, -50.9110107, 50.9974480
1: -198.8656006, 64.8961029, -182.0420074, 58.9841881, -257.8497925, 246.9381104
2: -107.0488434, 59.7006493, -97.2162018, 54.4071770, -161.4560242, 156.9168549
3: -138.4541931, 47.8215904, -126.5208359, 43.6295853, -182.0837708, 174.3424225
4: -75.6621323, 51.0305176, -68.7600708, 46.4838104, -122.1459427, 119.7905884

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.93 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -25.8702164, 27.7610855, -24.2387161, 25.9631062, -51.8333206, 51.9998016
1: -199.7743073, 65.1754684, -187.5152130, 61.0791168, -260.8533936, 252.6906738
2: -107.4355698, 60.0154800, -100.9924316, 56.1158714, -163.5514374, 161.0078735
3: -139.0041351, 48.0869255, -130.5455627, 45.0130157, -184.0171356, 178.6324921
4: -75.9317703, 51.3737907, -71.3170547, 47.9279099, -123.8596802, 122.6908417

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5221705, upper bound: 42.5228167
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5218475, upper bound: 42.5218475
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.74 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5222211, upper bound: 42.5229221
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5222210, upper bound: 42.5229219
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5221705, upper bound: 42.5228167
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -42.5218475, upper bound: 42.5218475

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -24.3851032, 26.2325764, -23.3464394, 25.1992130, -49.5843124, 49.5790100
1: -189.9980316, 61.4127235, -183.1970367, 58.9368248, -248.9348602, 244.6097412
2: -101.6896210, 56.5772514, -97.4684219, 54.5190926, -156.2087097, 154.0456696
3: -131.9941711, 45.4312897, -127.0827637, 43.7437973, -175.7379608, 172.5140381
4: -71.7767258, 48.3459244, -68.7357941, 46.5493774, -118.3261032, 117.0817032

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -23.7405701, 25.5392914, -23.6017952, 25.4609528, -49.2015190, 49.1410866
1: -185.9874878, 59.9482841, -185.1385345, 59.5990906, -245.5865631, 245.0868073
2: -99.6151657, 55.0895233, -98.6160583, 55.0716858, -154.6868439, 153.7055817
3: -129.2619781, 44.2586746, -128.5105896, 44.1956558, -173.4576263, 172.7692413
4: -70.1569061, 46.9472313, -69.5189819, 47.0025558, -117.1594620, 116.4662170

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23.4408360, 25.2006989, -23.8989925, 25.6601067, -49.1009293, 49.0996895
1: -182.9044037, 59.1448936, -186.0217896, 60.2173042, -243.1216583, 245.1666565
2: -98.0820618, 54.3497543, -99.8333893, 55.3368454, -153.4189148, 154.1831360
3: -127.1728210, 43.6682930, -129.3654022, 44.4523506, -171.6251678, 173.0336761
4: -69.1524963, 46.3824005, -70.4151382, 47.2407379, -116.3932190, 116.7975311

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5162269, upper bound: 42.5159998
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5136561, upper bound: 42.5139528
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -24.7744598, 26.8103523, -24.0014896, 25.8549366, -50.6293945, 50.8118439
1: -194.2111816, 62.5828323, -187.4286804, 60.5740051, -254.7851868, 250.0115051
2: -103.5835190, 57.8127975, -100.4100571, 55.7326775, -159.3161926, 158.2228546
3: -134.7849579, 46.3727722, -130.2604218, 44.7739983, -179.5589600, 176.6331635
4: -73.0605927, 49.4587746, -70.7865601, 47.6064262, -120.6670074, 120.2453308

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222210, upper bound: 42.5229219
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222210, upper bound: 42.5229219
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -26.0322247, 27.9063911, -23.0994244, 24.7951202, -50.8273468, 51.0058136
1: -200.7937775, 65.5595932, -178.9472198, 58.2086983, -259.0024719, 244.5068054
2: -108.0059738, 60.3809090, -95.8626480, 53.6869087, -161.6928558, 156.2435150
3: -139.7311096, 48.3582764, -124.4876328, 43.0255013, -182.7566071, 172.8459015
4: -76.3710632, 51.6670341, -67.8095322, 45.8775864, -122.2486496, 119.4765549

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.3244667, 27.1576900, -23.2113895, 24.9772816, -50.3017502, 50.3690758
1: -195.7739410, 63.8392258, -180.8399353, 58.4946785, -254.2686157, 244.6791534
2: -105.3205719, 58.7470398, -96.4671707, 53.9920578, -159.3126068, 155.2142029
3: -136.2613525, 47.0662727, -125.6546097, 43.2981491, -179.5595093, 172.7208405
4: -74.4259949, 50.2198906, -68.2084427, 46.1243706, -120.5503616, 118.4283218

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -25.2724781, 27.0868893, -23.8510437, 25.5248222, -50.7973022, 50.9379311
1: -195.0795441, 63.6806870, -184.4559326, 60.1047020, -255.1842346, 248.1365967
2: -105.0180664, 58.5782356, -99.4066238, 55.1844559, -160.2025146, 157.9848480
3: -135.7934265, 46.9436646, -128.4553070, 44.2483444, -180.0417786, 175.3989716
4: -74.2165833, 50.1128845, -70.1936798, 47.1170692, -121.3336487, 120.3065567

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5164637, upper bound: 42.5161258
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5138826, upper bound: 42.5141030
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -26.0387230, 28.0663071, -23.8000050, 25.5490589, -51.5877838, 51.8663063
1: -201.5428162, 65.6903076, -184.3120880, 59.9874039, -261.5302124, 250.0023651
2: -108.0362091, 60.6945114, -99.0699768, 55.1377220, -163.1739044, 159.7644501
3: -140.1417542, 48.5750618, -128.2044220, 44.2472839, -184.3890381, 176.7794800
4: -76.4665146, 52.0326195, -70.0551529, 47.1398544, -123.6063690, 122.0877533

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217470, upper bound: 42.5213762
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5212747, upper bound: 42.5212747
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5162269, upper bound: 42.5159998
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5136561, upper bound: 42.5139528
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5222210, upper bound: 42.5229219
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5222210, upper bound: 42.5229219
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5164637, upper bound: 42.5161258
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5138826, upper bound: 42.5141030
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5217470, upper bound: 42.5213762
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 0, lower bound: -42.5212747, upper bound: 42.5212747

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -24.3851032, 26.2325764, -22.8997803, 24.7557316, -49.1408348, 49.1323547
1: -189.9980316, 61.4127235, -180.5335999, 57.7892418, -247.7872772, 241.9463196
2: -101.6896210, 56.5772514, -95.6624985, 53.5586014, -155.2482147, 152.2397308
3: -131.9941711, 45.4312897, -124.9866791, 42.9730034, -174.9671783, 170.4179688
4: -71.7767258, 48.3459244, -67.4286346, 45.7246704, -117.5013962, 115.7745361

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -24.3851032, 26.2325764, -24.3518753, 26.1530285, -50.5381165, 50.5844498
1: -189.9980316, 61.4127235, -187.5906067, 61.2857590, -251.2837830, 249.0033112
2: -101.6896210, 56.5772514, -100.5393982, 56.6837273, -158.3733521, 157.1166077
3: -131.9941711, 45.4312897, -130.4767609, 45.3737221, -177.3678894, 175.9080505
4: -71.7767258, 48.3459244, -71.1771393, 48.5826836, -120.3594055, 119.5230408

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -23.7405701, 25.5392914, -23.2374191, 25.0616093, -48.8021736, 48.7767105
1: -185.9874878, 59.9482841, -182.6204681, 58.6245918, -244.6120605, 242.5687561
2: -99.6151657, 55.0895233, -97.0546646, 54.2143059, -153.8294525, 152.1441956
3: -129.2619781, 44.2586746, -126.5950241, 43.5095482, -172.7715302, 170.8536987
4: -70.1569061, 46.9472313, -68.4261017, 46.2690926, -116.4259949, 115.3733368

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -23.7405701, 25.5392914, -24.4617348, 26.3309746, -50.0715408, 50.0010262
1: -185.9874878, 59.9482841, -189.2243652, 61.5799789, -247.5674591, 249.1726532
2: -99.6151657, 55.0895233, -101.1078262, 56.9714622, -156.5866089, 156.1973572
3: -129.2619781, 44.2586746, -131.4463959, 45.6184425, -174.8804169, 175.7050781
4: -70.1569061, 46.9472313, -71.5705948, 48.8309402, -118.9878464, 118.5178223

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -24.7744598, 26.8103523, -23.5755234, 25.4189663, -50.1934280, 50.3858719
1: -194.2111816, 62.5828323, -184.4756775, 59.5148811, -253.7260590, 247.0585022
2: -103.5835190, 57.8127975, -98.6994171, 54.7765198, -158.3600006, 156.5122070
3: -134.7849579, 46.3727722, -128.1447906, 44.0196114, -178.8045654, 174.5175323
4: -73.0605927, 49.4587746, -69.5540237, 46.7882576, -119.8488388, 119.0128021

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5216325, upper bound: 42.5223500
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -24.7744598, 26.8103523, -25.0891838, 26.9426231, -51.7170830, 51.8995361
1: -194.2111816, 62.5828323, -193.1970520, 63.1986885, -257.4098511, 255.7798767
2: -103.5835190, 57.8127975, -103.9697723, 58.2189369, -161.8024445, 161.7825470
3: -134.7849579, 46.3727722, -134.4605865, 46.6655731, -181.4505310, 180.8333282
4: -73.0605927, 49.4587746, -73.5701828, 49.8991547, -122.9597321, 123.0289612

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5216325, upper bound: 42.5223500
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -26.0322247, 27.9063911, -22.8997803, 24.7557316, -50.7879562, 50.8061714
1: -200.7937775, 65.5595932, -180.5335999, 57.7892418, -258.5830078, 246.0932007
2: -108.0059738, 60.3809090, -95.6624985, 53.5586014, -161.5645599, 156.0433655
3: -139.7311096, 48.3582764, -124.9866791, 42.9730034, -182.7041168, 173.3449554
4: -76.3710632, 51.6670341, -67.4286346, 45.7246704, -122.0957336, 119.0956573

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -26.0322247, 27.9063911, -24.3518753, 26.1530285, -52.1852493, 52.2582664
1: -200.7937775, 65.5595932, -187.5906067, 61.2857590, -262.0795288, 253.1501923
2: -108.0059738, 60.3809090, -100.5393982, 56.6837273, -164.6896973, 160.9202576
3: -139.7311096, 48.3582764, -130.4767609, 45.3737221, -185.1048279, 178.8350372
4: -76.3710632, 51.6670341, -71.1771393, 48.5826836, -124.9537506, 122.8441620

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.3244667, 27.1576900, -23.2371941, 25.0613499, -50.3858070, 50.3948708
1: -195.7739410, 63.8392258, -182.6185913, 58.6239357, -254.3978729, 246.4578094
2: -105.3205719, 58.7470398, -97.0536194, 54.2137489, -159.5342865, 155.8006439
3: -136.2613525, 47.0662727, -126.5936356, 43.5090866, -179.7704315, 173.6598816
4: -74.4259949, 50.2198906, -68.4253540, 46.2686348, -120.6946259, 118.6452484

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.3244667, 27.1576900, -24.4617348, 26.3309746, -51.6554375, 51.6194229
1: -195.7739410, 63.8392258, -189.2243652, 61.5799789, -257.3539124, 253.0635986
2: -105.3205719, 58.7470398, -101.1078262, 56.9714622, -162.2920380, 159.8548584
3: -136.2613525, 47.0662727, -131.4463959, 45.6184425, -181.8797913, 178.5126648
4: -74.4259949, 50.2198906, -71.5705948, 48.8309402, -123.2569351, 121.7904739

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -25.8284245, 27.8300953, -23.3438148, 25.0374050, -50.8658257, 51.1739120
1: -199.8988953, 65.1615829, -180.7229614, 58.8451729, -258.7440186, 245.8845520
2: -107.2006226, 60.1893463, -97.2647018, 54.0537300, -161.2543182, 157.4540405
3: -139.0257111, 48.1655006, -125.7839432, 43.3679924, -182.3936615, 173.9494476
4: -75.8693542, 51.5802841, -68.7589798, 46.1655159, -122.0348663, 120.3392639

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5212747, upper bound: 42.5212747
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5212747, upper bound: 42.5212747
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.04 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5216325, upper bound: 42.5223500
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5216325, upper bound: 42.5223500
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5212747, upper bound: 42.5212747
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.04
Output dim: 0, lower bound: -42.5212747, upper bound: 42.5212747

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23.6550789, 25.6027508, -22.8997803, 24.7557316, -48.4108047, 48.5025330
1: -186.2450104, 59.6139526, -180.5335999, 57.7892418, -244.0342407, 240.1475525
2: -98.5096741, 55.3716087, -95.6624985, 53.5586014, -152.0682678, 151.0340881
3: -128.8228302, 44.4017982, -124.9866791, 42.9730034, -171.7958221, 169.3884735
4: -69.4679184, 47.3655510, -67.4286346, 45.7246704, -115.1925888, 114.7941895

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5245329
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5248527
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -24.2303600, 26.0971661, -22.8997803, 24.7557316, -48.9860878, 48.9969482
1: -188.8207855, 60.9978714, -180.5335999, 57.7892418, -246.6100311, 241.5314636
2: -100.9717255, 56.2269554, -95.6624985, 53.5586014, -154.5303345, 151.8894196
3: -131.0939941, 45.1809883, -124.9866791, 42.9730034, -174.0669861, 170.1676636
4: -71.2263718, 48.1115761, -67.4286346, 45.7246704, -116.9510422, 115.5402069

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5245329
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5248753
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23.6550789, 25.6027508, -24.3518753, 26.1530285, -49.8080864, 49.9546280
1: -186.2450104, 59.6139526, -187.5906067, 61.2857590, -247.5307465, 247.2045441
2: -98.5096741, 55.3716087, -100.5393982, 56.6837273, -155.1934052, 155.9109802
3: -128.8228302, 44.4017982, -130.4767609, 45.3737221, -174.1965332, 174.8785553
4: -69.4679184, 47.3655510, -71.1771393, 48.5826836, -118.0505981, 118.5426941

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -24.2303600, 26.0971661, -24.3518753, 26.1530285, -50.3833656, 50.4490395
1: -188.8207855, 60.9978714, -187.5906067, 61.2857590, -250.1065369, 248.5884705
2: -100.9717255, 56.2269554, -100.5393982, 56.6837273, -157.6554565, 156.7662964
3: -131.0939941, 45.1809883, -130.4767609, 45.3737221, -176.4676819, 175.6577454
4: -71.2263718, 48.1115761, -71.1771393, 48.5826836, -119.8090439, 119.2887115

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -23.1128292, 24.9189796, -23.2374191, 25.0616093, -48.1744347, 48.1563950
1: -181.6424103, 58.3029137, -182.6204681, 58.6245918, -240.2669525, 240.9233856
2: -96.5479965, 53.9097023, -97.0546646, 54.2143059, -150.7622986, 150.9643707
3: -125.9141998, 43.2622185, -126.5950241, 43.5095482, -169.4237366, 169.8572388
4: -68.0682220, 46.0008202, -68.4261017, 46.2690926, -114.3373108, 114.4269180

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253552, upper bound: 42.5253107
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5254830, upper bound: 42.5254028
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -23.6116467, 25.4272537, -23.2374191, 25.0616093, -48.6732521, 48.6646729
1: -185.0122681, 59.5900726, -182.6204681, 58.6245918, -243.6368256, 242.2105103
2: -98.9901123, 54.7915535, -97.0546646, 54.2143059, -153.2044067, 151.8462219
3: -128.5012360, 44.0521011, -126.5950241, 43.5095482, -172.0107880, 170.6471100
4: -69.6954880, 46.7554893, -68.4261017, 46.2690926, -115.9645844, 115.1815872

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253552, upper bound: 42.5253107
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5254830, upper bound: 42.5254028
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -23.1128292, 24.9189796, -24.4617348, 26.3309746, -49.4437981, 49.3807144
1: -181.6424103, 58.3029137, -189.2243652, 61.5799789, -243.2223511, 247.5272827
2: -96.5479965, 53.9097023, -101.1078262, 56.9714622, -153.5194550, 155.0175323
3: -125.9141998, 43.2622185, -131.4463959, 45.6184425, -171.5326385, 174.7086182
4: -68.0682220, 46.0008202, -71.5705948, 48.8309402, -116.8991623, 117.5714111

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -23.6116467, 25.4272537, -24.4617348, 26.3309746, -49.9426155, 49.8889885
1: -185.0122681, 59.5900726, -189.2243652, 61.5799789, -246.5922241, 248.8144073
2: -98.9901123, 54.7915535, -101.1078262, 56.9714622, -155.9615784, 155.8993835
3: -128.5012360, 44.0521011, -131.4463959, 45.6184425, -174.1196747, 175.4985046
4: -69.6954880, 46.7554893, -71.5705948, 48.8309402, -118.5264282, 118.3260803

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -24.3296223, 26.3126507, -23.3437557, 25.1613159, -49.4909363, 49.6564064
1: -190.7111206, 61.4653015, -182.6727600, 58.9312820, -249.6423798, 244.1380463
2: -101.8072357, 56.7499657, -97.7788162, 54.2217712, -156.0290070, 154.5287781
3: -132.4071350, 45.5152054, -126.9164429, 43.5736046, -175.9807129, 172.4316406
4: -71.7928848, 48.5084267, -68.8965759, 46.2942429, -118.0871277, 117.4049988

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5208906, upper bound: 42.5216289
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249854, upper bound: 42.5250093
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -24.9506645, 26.9262905, -22.9786758, 24.7518597, -49.7025223, 49.9049644
1: -193.1208801, 62.8909760, -179.5338135, 57.9953651, -251.1162415, 242.4247894
2: -103.5651550, 57.9651031, -96.1963043, 53.3154602, -156.8806000, 154.1614075
3: -134.2054596, 46.5282516, -124.7587509, 42.8698387, -177.0752869, 171.2869873
4: -73.4086533, 49.7681198, -67.8230286, 45.5332298, -118.9418793, 117.5911484

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5207894, upper bound: 42.5213941
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249868, upper bound: 42.5249868
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -24.3296223, 26.3126507, -24.8408566, 26.6684093, -50.9980316, 51.1535072
1: -190.7111206, 61.4653015, -191.2686005, 62.5714645, -253.2825623, 252.7339020
2: -101.8072357, 56.7499657, -102.9812622, 57.6243515, -159.4315796, 159.7312317
3: -132.4071350, 45.5152054, -133.1467285, 46.1861725, -178.5933075, 178.6619263
4: -71.7928848, 48.5084267, -72.8640594, 49.3730621, -121.1659317, 121.3724823

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -24.9506645, 26.9262905, -24.5310726, 26.3226910, -51.2733536, 51.4573631
1: -193.1208801, 62.8909760, -188.5119019, 61.7715225, -254.8923798, 251.4028778
2: -103.5651550, 57.9651031, -101.5916595, 56.8443222, -160.4094696, 159.5567627
3: -134.2054596, 46.5282516, -131.2316284, 45.5904427, -179.7958527, 177.7598572
4: -73.4086533, 49.7681198, -71.9382553, 48.7303009, -122.1389465, 121.7063751

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5155672, upper bound: 42.5157169
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5085759, upper bound: 42.5091863
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -25.0333824, 26.9096222, -22.8997803, 24.7557316, -49.7891159, 49.8094025
1: -192.9763031, 62.9468193, -180.5335999, 57.7892418, -250.7655487, 243.4804230
2: -103.1665802, 58.3487778, -95.6624985, 53.5586014, -156.7251892, 154.0112610
3: -134.0050354, 46.6901398, -124.9866791, 42.9730034, -176.9780426, 171.6768188
4: -73.0677185, 50.0716019, -67.4286346, 45.7246704, -118.7923889, 117.5002289

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245901, upper bound: 42.5238166
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240867, upper bound: 42.5234323
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -25.7930584, 27.6743031, -22.8997803, 24.7557316, -50.5487900, 50.5740814
1: -198.7024384, 64.8984756, -180.5335999, 57.7892418, -256.4916687, 245.4320679
2: -106.8090286, 59.8184204, -95.6624985, 53.5586014, -160.3676147, 155.4808807
3: -138.1825562, 47.9299545, -124.9866791, 42.9730034, -181.1555634, 172.9166260
4: -75.5250473, 51.2789764, -67.4286346, 45.7246704, -121.2497177, 118.7076035

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5245901, upper bound: 42.5241415
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240867, upper bound: 42.5238233
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -25.0333824, 26.9096222, -24.3518753, 26.1530285, -51.1863976, 51.2614975
1: -192.9763031, 62.9468193, -187.5906067, 61.2857590, -254.2620544, 250.5373993
2: -103.1665802, 58.3487778, -100.5393982, 56.6837273, -159.8503113, 158.8881683
3: -134.0050354, 46.6901398, -130.4767609, 45.3737221, -179.3787537, 177.1669006
4: -73.0677185, 50.0716019, -71.1771393, 48.5826836, -121.6504059, 121.2487335

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -25.7930584, 27.6743031, -24.3518753, 26.1530285, -51.9460716, 52.0261765
1: -198.7024384, 64.8984756, -187.5906067, 61.2857590, -259.9881897, 252.4890747
2: -106.8090286, 59.8184204, -100.5393982, 56.6837273, -163.4927521, 160.3577881
3: -138.1825562, 47.9299545, -130.4767609, 45.3737221, -183.5562744, 178.4067078
4: -75.5250473, 51.2789764, -71.1771393, 48.5826836, -124.1077271, 122.4561081

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -24.2544250, 26.1291466, -23.2371941, 25.0613499, -49.3157616, 49.3663330
1: -187.9201508, 61.0501823, -182.6185913, 58.6239357, -246.5440826, 243.6687775
2: -100.2535095, 56.5268555, -97.0536194, 54.2137489, -154.4672546, 153.5804596
3: -130.4596405, 45.2624550, -126.5936356, 43.5090866, -173.9687195, 171.8560944
4: -70.9664917, 48.4487038, -68.4253540, 46.2686348, -117.2351227, 116.8740540

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.1207199, 26.9715672, -23.2371941, 25.0613499, -50.1820526, 50.2087479
1: -194.0323486, 63.2702599, -182.6185913, 58.6239357, -252.6562805, 245.8888550
2: -104.2892532, 58.2751694, -97.0536194, 54.2137489, -158.5029755, 155.3287659
3: -134.9489288, 46.7267418, -126.5936356, 43.5090866, -178.4579926, 173.3203735
4: -73.6893845, 49.9094925, -68.4253540, 46.2686348, -119.9580231, 118.3348465

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -24.2544250, 26.1291466, -24.4617348, 26.3309746, -50.5853920, 50.5908813
1: -187.9201508, 61.0501823, -189.2243652, 61.5799789, -249.5001068, 250.2745514
2: -100.2535095, 56.5268555, -101.1078262, 56.9714622, -157.2249756, 157.6346741
3: -130.4596405, 45.2624550, -131.4463959, 45.6184425, -176.0780792, 176.7088470
4: -70.9664917, 48.4487038, -71.5705948, 48.8309402, -119.7974319, 120.0193024

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -25.1207199, 26.9715672, -24.4617348, 26.3309746, -51.4516830, 51.4333000
1: -194.0323486, 63.2702599, -189.2243652, 61.5799789, -255.6123199, 252.4946136
2: -104.2892532, 58.2751694, -101.1078262, 56.9714622, -161.2606964, 159.3829803
3: -134.9489288, 46.7267418, -131.4463959, 45.6184425, -180.5673676, 178.1731415
4: -73.6893845, 49.9094925, -71.5705948, 48.8309402, -122.5203247, 121.4800797

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.92 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5245329
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5248527
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5245329
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5247295, upper bound: 42.5248753
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236754, upper bound: 42.5237510
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5253552, upper bound: 42.5253107
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5254830, upper bound: 42.5254028
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5253552, upper bound: 42.5253107
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5254830, upper bound: 42.5254028
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237336, upper bound: 42.5238837
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5208906, upper bound: 42.5216289
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5249854, upper bound: 42.5250093
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5207894, upper bound: 42.5213941
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5249868, upper bound: 42.5249868
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5217453, upper bound: 42.5227928
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5155672, upper bound: 42.5157169
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5085759, upper bound: 42.5091863
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5245901, upper bound: 42.5238166
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5240867, upper bound: 42.5234323
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5245901, upper bound: 42.5241415
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5240867, upper bound: 42.5238233
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5236806, upper bound: 42.5236394
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5237630, upper bound: 42.5234937
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -42.5235235, upper bound: 42.5232791

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -22.8908653, 24.7620411, -21.7871246, 23.5441780, -46.4350395, 46.5491638
1: -179.7460175, 57.7596741, -171.4611206, 55.0663757, -234.8123779, 229.2207947
2: -95.4293823, 53.5218239, -91.2484360, 50.8093872, -146.2387695, 144.7702637
3: -124.5495834, 42.9634933, -119.0146332, 40.8444557, -165.3940430, 161.9781189
4: -67.2967834, 45.7842445, -64.2702560, 43.3705635, -110.6673355, 110.0544968

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241836, upper bound: 42.5242219
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241560, upper bound: 42.5239917
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -23.2709904, 25.2492409, -24.1302643, 26.2390652, -49.5100555, 49.3795013
1: -184.0401306, 58.6255722, -191.1051636, 61.0178108, -245.0579376, 249.7307129
2: -97.0426788, 54.5564728, -100.7518692, 56.7179642, -153.7606506, 155.3083496
3: -127.1542892, 43.7318649, -132.2690582, 45.4960556, -172.6503448, 176.0009155
4: -68.2137604, 46.6771202, -71.0384979, 48.4461060, -116.6598663, 117.7156143

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241718, upper bound: 42.5244216
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241955, upper bound: 42.5241426
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -23.6029034, 25.4073582, -21.7871246, 23.5441780, -47.1470680, 47.1944809
1: -183.8869476, 59.4075470, -171.4611206, 55.0663757, -238.9533234, 230.8686523
2: -98.4060974, 54.7249947, -91.2484360, 50.8093872, -149.2154846, 145.9734344
3: -127.7004242, 43.9844742, -119.0146332, 40.8444557, -168.5448761, 162.9991150
4: -69.4164581, 46.8099098, -64.2702560, 43.3705635, -112.7870178, 111.0801544

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241946, upper bound: 42.5242003
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5238635
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -23.6111488, 25.5381622, -24.1302643, 26.2390652, -49.8502121, 49.6684265
1: -184.7021637, 59.4903297, -191.1051636, 61.0178108, -245.7199554, 250.5954742
2: -98.4581528, 54.9354630, -100.7518692, 56.7179642, -155.1760864, 155.6873322
3: -128.1031342, 44.1666260, -132.2690582, 45.4960556, -173.5991669, 176.4356842
4: -69.4308167, 47.0631828, -71.0384979, 48.4461060, -117.8769226, 118.1016846

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241850, upper bound: 42.5244460
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5241207
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -23.6550789, 25.6027508, -25.0334702, 26.9097290, -50.5647964, 50.6362228
1: -186.2450104, 59.6139526, -192.9769745, 62.9470634, -249.1920471, 252.5908813
2: -98.5096741, 55.3716087, -103.1669617, 58.3489990, -156.8586731, 158.5385742
3: -128.8228302, 44.4017982, -134.0055237, 46.6903191, -175.5131378, 178.4072876
4: -69.4679184, 47.3655510, -73.0679855, 50.0717926, -119.5397110, 120.4335327

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5206470, upper bound: 42.5214751
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5199634, upper bound: 42.5201090
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -23.6550789, 25.6027508, -24.2544250, 26.1291466, -49.7842216, 49.8571701
1: -186.2450104, 59.6139526, -187.9201508, 61.0501823, -247.2951813, 247.5340881
2: -98.5096741, 55.3716087, -100.2535095, 56.5268555, -155.0365295, 155.6251221
3: -128.8228302, 44.4017982, -130.4596405, 45.2624550, -174.0852814, 174.8614349
4: -69.4679184, 47.3655510, -70.9664917, 48.4487038, -117.9166260, 118.3320465

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5206470, upper bound: 42.5214751
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5199634, upper bound: 42.5201090
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -24.2303600, 26.0971661, -25.0334702, 26.9097290, -51.1400757, 51.1306343
1: -188.8207855, 60.9978714, -192.9769745, 62.9470634, -251.7678375, 253.9748383
2: -100.9717255, 56.2269554, -103.1669617, 58.3489990, -159.3207245, 159.3939056
3: -131.0939941, 45.1809883, -134.0055237, 46.6903191, -177.7842865, 179.1865082
4: -71.2263718, 48.1115761, -73.0679855, 50.0717926, -121.2981567, 121.1795654

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5206250, upper bound: 42.5213497
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5201857, upper bound: 42.5202201
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -24.2303600, 26.0971661, -24.2544250, 26.1291466, -50.3595047, 50.3515816
1: -188.8207855, 60.9978714, -187.9201508, 61.0501823, -249.8709564, 248.9180145
2: -100.9717255, 56.2269554, -100.2535095, 56.5268555, -157.4985809, 156.4804688
3: -131.0939941, 45.1809883, -130.4596405, 45.2624550, -176.3564453, 175.6406250
4: -71.2263718, 48.1115761, -70.9664917, 48.4487038, -119.6750717, 119.0780640

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5206250, upper bound: 42.5213497
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5201857, upper bound: 42.5202201
time: 1.44 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -21.8403378, 23.5594635, -20.9822063, 22.5920029, -44.4323425, 44.5416679
1: -172.9338531, 55.0090828, -166.0469666, 52.8208084, -225.7546234, 221.0560455
2: -91.5236588, 50.9636040, -88.0630722, 48.8868561, -140.4105225, 139.0266724
3: -119.6245193, 40.8660965, -114.9114532, 39.2117500, -158.8362732, 155.7775574
4: -64.1298218, 43.4070244, -61.6250153, 41.5940857, -105.7239075, 105.0320206

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243707, upper bound: 42.5248637
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243230, upper bound: 42.5246129
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -22.3137741, 24.0818024, -22.0892792, 23.8606949, -46.1744690, 46.1710739
1: -175.5457764, 56.3951035, -174.1327820, 55.8543625, -231.4000854, 230.5278473
2: -93.4725266, 52.0731163, -92.6196365, 51.5620842, -145.0346069, 144.6927490
3: -121.8734283, 41.8046303, -120.8453751, 41.4022369, -163.2756653, 162.6500092
4: -65.8493500, 44.4077034, -65.2307892, 43.9638367, -109.8131866, 109.6384888

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5254734, upper bound: 42.5255361
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5253425, upper bound: 42.5255361
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -21.8083820, 23.5295391, -20.9822063, 22.5920029, -44.4003830, 44.5117455
1: -172.1354675, 55.1334305, -166.0469666, 52.8208084, -224.9562531, 221.1803894
2: -91.7228851, 50.7301064, -88.0630722, 48.8868561, -140.6097412, 138.7931519
3: -119.4341431, 40.8055878, -114.9114532, 39.2117500, -158.6458893, 155.7170410
4: -64.4980774, 43.2215309, -61.6250153, 41.5940857, -106.0921631, 104.8465195

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243756, upper bound: 42.5247657
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5242121
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -23.1132317, 24.8969498, -22.0892792, 23.8606949, -46.9739189, 46.9862175
1: -181.2551270, 58.3470688, -174.1327820, 55.8543625, -237.1094666, 232.4798431
2: -96.9931564, 53.6417084, -92.6196365, 51.5620842, -148.5552368, 146.2613525
3: -125.8859177, 43.1326485, -120.8453751, 41.4022369, -167.2881470, 163.9780273
4: -68.2683029, 45.7647591, -65.2307892, 43.9638367, -112.2321396, 110.9955444

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249431, upper bound: 42.5249454
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5245018
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -23.1128292, 24.9189796, -25.0334435, 26.9096889, -50.0225105, 49.9524193
1: -181.6424103, 58.3029137, -192.9767303, 62.9469643, -244.5893555, 251.2796478
2: -96.5479965, 53.9097023, -103.1668091, 58.3489227, -154.8969116, 157.0764923
3: -125.9141998, 43.2622185, -134.0053101, 46.6902618, -172.6044617, 177.2675323
4: -68.0682220, 46.0008202, -73.0678787, 50.0717163, -118.1399384, 119.0686951

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5206889, upper bound: 42.5215553
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5199898, upper bound: 42.5201819
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -23.1128292, 24.9189796, -24.2544250, 26.1291466, -49.2419739, 49.1733971
1: -181.6424103, 58.3029137, -187.9201508, 61.0501823, -242.6925964, 246.2230682
2: -96.5479965, 53.9097023, -100.2535095, 56.5268555, -153.0748596, 154.1632080
3: -125.9141998, 43.2622185, -130.4596405, 45.2624550, -171.1766510, 173.7218628
4: -68.0682220, 46.0008202, -70.9664917, 48.4487038, -116.5169220, 116.9673157

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5206889, upper bound: 42.5215553
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5199898, upper bound: 42.5201819
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -23.6116467, 25.4272537, -25.0334435, 26.9096889, -50.5213280, 50.4606895
1: -185.0122681, 59.5900726, -192.9767303, 62.9469643, -247.9592285, 252.5667877
2: -98.9901123, 54.7915535, -103.1668091, 58.3489227, -157.3390198, 157.9583588
3: -128.5012360, 44.0521011, -134.0053101, 46.6902618, -175.1914978, 178.0574036
4: -69.6954880, 46.7554893, -73.0678787, 50.0717163, -119.7672043, 119.8233643

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5207980, upper bound: 42.5218156
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5191768, upper bound: 42.5190236
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -23.6116467, 25.4272537, -24.2544250, 26.1291466, -49.7407913, 49.6816711
1: -185.0122681, 59.5900726, -187.9201508, 61.0501823, -246.0624542, 247.5101929
2: -98.9901123, 54.7915535, -100.2535095, 56.5268555, -155.5169678, 155.0450592
3: -128.5012360, 44.0521011, -130.4596405, 45.2624550, -173.7636871, 174.5117493
4: -69.6954880, 46.7554893, -70.9664917, 48.4487038, -118.1441956, 117.7219849

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5207980, upper bound: 42.5218156
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5191768, upper bound: 42.5190236
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -23.8083038, 25.7378120, -22.3748875, 24.1033363, -47.9116402, 48.1127014
1: -186.5358276, 60.1446190, -175.1475983, 56.4942131, -243.0300446, 235.2922211
2: -99.6661758, 55.4975586, -93.8512421, 51.9231262, -151.5892944, 149.3488007
3: -129.5444794, 44.5192680, -121.7321625, 41.7428780, -171.2873383, 166.2514343
4: -70.2756119, 47.4310417, -66.1024933, 44.2985458, -114.5741501, 113.5335388

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5168892, upper bound: 42.5202806
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5193757, upper bound: 42.5206578
time: 1.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -23.7564983, 25.8078747, -24.5315037, 26.6411076, -50.3976021, 50.3393784
1: -187.1028748, 60.0816536, -193.5557709, 62.0990524, -249.2019348, 253.6373901
2: -99.5277863, 55.5772705, -102.6966934, 57.4664154, -156.9942017, 158.2739563
3: -129.7696991, 44.5911217, -134.1669006, 46.0979309, -175.8675842, 178.7580261
4: -70.1714554, 47.5465050, -72.4242172, 49.0964050, -119.2678452, 119.9707108

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246219, upper bound: 42.5248602
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246137, upper bound: 42.5247623
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -24.3044415, 26.3497868, -24.1403542, 26.2135315, -50.5179749, 50.4901390
1: -189.1340790, 61.3286285, -190.2564697, 61.1008148, -250.2348938, 251.5850830
2: -101.0512390, 56.6414299, -101.0303421, 56.5064545, -157.5576630, 157.6717377
3: -131.2911835, 45.4819107, -131.8913116, 45.3492241, -176.6403809, 177.3732147
4: -71.5871582, 48.6663513, -71.2895203, 48.2948303, -119.8819809, 119.9558563

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5246219, upper bound: 42.5248320
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5203329, upper bound: 42.5246137
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -24.3296223, 26.3126507, -24.7044525, 26.4689980, -50.7986221, 51.0171051
1: -190.7111206, 61.4653015, -190.5687408, 62.2247581, -252.9358521, 252.0340424
2: -101.8072357, 56.7499657, -102.6396713, 57.2202148, -159.0274506, 159.3896332
3: -132.4071350, 45.5152054, -132.6580658, 45.8635521, -178.2706909, 178.1732788
4: -71.7928848, 48.5084267, -72.5263748, 48.9553299, -120.7482147, 121.0347824

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5187934, upper bound: 42.5194972
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5181255, upper bound: 42.5188300
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5181279, upper bound: 42.5179985
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217328, upper bound: 42.5227806
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -24.3296223, 26.3126507, -25.4913998, 27.4659500, -51.7955666, 51.8040504
1: -190.7111206, 61.4653015, -197.2038879, 64.2941818, -255.0052643, 258.6691895
2: -101.8072357, 56.7499657, -105.7585144, 59.3844604, -161.1916962, 162.5084839
3: -132.4071350, 45.5152054, -137.1338043, 47.5339699, -179.9411011, 182.6490021
4: -71.7928848, 48.5084267, -74.8460541, 50.9084549, -122.7013397, 123.3544769

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5187934, upper bound: 42.5220848
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5181255, upper bound: 42.5188300
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5181279, upper bound: 42.5180203
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217328, upper bound: 42.5227806
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -24.3748226, 26.1686649, -21.6534424, 23.2101631, -47.5849838, 47.8221054
1: -187.6248322, 61.3050423, -169.8864441, 54.4691811, -242.0939789, 231.1914825
2: -100.5149307, 56.7411766, -90.7511826, 50.2763824, -150.7913055, 147.4923401
3: -130.4643097, 45.4112053, -117.7960892, 40.2697487, -170.7340546, 163.2072906
4: -71.1750183, 48.6529579, -63.6434860, 42.8048325, -113.9798508, 112.2964478

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5167227, upper bound: 42.5159211
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5079494, upper bound: 42.5071085
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -24.9050026, 26.7706985, -22.6199951, 24.4527454, -49.3577499, 49.3906937
1: -192.0466309, 62.6204224, -178.4935455, 57.0847816, -249.1314087, 241.1139679
2: -102.6401596, 58.0477219, -94.5390015, 52.9038200, -155.5439758, 152.5867157
3: -133.3355560, 46.4497490, -123.5418549, 42.4491348, -175.7846985, 169.9915924
4: -72.6970291, 49.8106995, -66.6277695, 45.1491165, -117.8461456, 116.4384613

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5074790, upper bound: 42.5080498
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5033994, upper bound: 42.5035118
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.1934719, 27.0002708, -21.6534424, 23.2101631, -48.4036331, 48.6537132
1: -193.9222412, 63.3822708, -169.8864441, 54.4691811, -248.3913727, 233.2687073
2: -104.3593369, 58.3664780, -90.7511826, 50.2763824, -154.6357117, 149.1176605
3: -134.9162903, 46.7694969, -117.7960892, 40.2697487, -175.1860199, 164.5655823
4: -73.7986603, 50.0082817, -63.6434860, 42.8048325, -116.6034927, 113.6517639

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228079, upper bound: 42.5230474
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222818, upper bound: 42.5217739
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.6752739, 27.5449963, -22.6199951, 24.4527454, -50.1280098, 50.1649933
1: -197.7905273, 64.6033249, -178.4935455, 57.0847816, -254.8753052, 243.0968475
2: -106.3299866, 59.5386467, -94.5390015, 52.9038200, -159.2338104, 154.0776520
3: -137.5544434, 47.7070122, -123.5418549, 42.4491348, -180.0035706, 171.2488556
4: -75.1871567, 51.0357132, -66.6277695, 45.1491165, -120.3362656, 117.6634827

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5226094, upper bound: 42.5230374
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5220130, upper bound: 42.5215269
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -25.0333824, 26.9096222, -25.0334702, 26.9097290, -51.9431038, 51.9430923
1: -192.9763031, 62.9468193, -192.9769745, 62.9470634, -255.9233704, 255.9237823
2: -103.1665802, 58.3487778, -103.1669617, 58.3489990, -161.5155792, 161.5157471
3: -134.0050354, 46.6901398, -134.0055237, 46.6903191, -180.6953583, 180.6956177
4: -73.0677185, 50.0716019, -73.0679855, 50.0717926, -123.1395111, 123.1395874

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5224670, upper bound: 42.5221629
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217812, upper bound: 42.5217364
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -25.0333824, 26.9096222, -24.2544250, 26.1291466, -51.1625290, 51.1640434
1: -192.9763031, 62.9468193, -187.9201508, 61.0501823, -254.0264893, 250.8669586
2: -103.1665802, 58.3487778, -100.2535095, 56.5268555, -159.6934357, 158.6022949
3: -134.0050354, 46.6901398, -130.4596405, 45.2624550, -179.2674866, 177.1497803
4: -73.0677185, 50.0716019, -70.9664917, 48.4487038, -121.5164185, 121.0380936

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5224670, upper bound: 42.5221832
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217812, upper bound: 42.5217933
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -25.7930584, 27.6743031, -25.0334702, 26.9097290, -52.7027817, 52.7077713
1: -198.7024384, 64.8984756, -192.9769745, 62.9470634, -261.6494751, 257.8754578
2: -106.8090286, 59.8184204, -103.1669617, 58.3489990, -165.1580200, 162.9853821
3: -138.1825562, 47.9299545, -134.0055237, 46.6903191, -184.8728638, 181.9354858
4: -75.5250473, 51.2789764, -73.0679855, 50.0717926, -125.5968399, 124.3469620

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5204939, upper bound: 42.5213028
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5187680, upper bound: 42.5186794
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -25.7930584, 27.6743031, -24.2544250, 26.1291466, -51.9222031, 51.9287224
1: -198.7024384, 64.8984756, -187.9201508, 61.0501823, -259.7525940, 252.8186340
2: -106.8090286, 59.8184204, -100.2535095, 56.5268555, -163.3358612, 160.0719299
3: -138.1825562, 47.9299545, -130.4596405, 45.2624550, -183.4450073, 178.3895874
4: -75.5250473, 51.2789764, -70.9664917, 48.4487038, -123.9737549, 122.2454681

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5204939, upper bound: 42.5213028
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5187680, upper bound: 42.5186794
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -24.2544250, 26.1291466, -23.6550789, 25.6027508, -49.8571701, 49.7842216
1: -187.9201508, 61.0501823, -186.2450104, 59.6139526, -247.5340729, 247.2951813
2: -100.2535095, 56.5268555, -98.5096741, 55.3716087, -155.6251221, 155.0365295
3: -130.4596405, 45.2624550, -128.8228302, 44.4017982, -174.8614349, 174.0852814
4: -70.9664917, 48.4487038, -69.4679184, 47.3655510, -118.3320465, 117.9166260

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228517, upper bound: 42.5230702
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5227526, upper bound: 42.5226332
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -24.2544250, 26.1291466, -23.1128292, 24.9189796, -49.1734009, 49.2419739
1: -187.9201508, 61.0501823, -181.6424103, 58.3029137, -246.2230682, 242.6925964
2: -100.2535095, 56.5268555, -96.5479965, 53.9097023, -154.1632080, 153.0748596
3: -130.4596405, 45.2624550, -125.9141998, 43.2622185, -173.7218628, 171.1766510
4: -70.9664917, 48.4487038, -68.0682220, 46.0008202, -116.9673157, 116.5169220

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228517, upper bound: 42.5230702
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5227526, upper bound: 42.5226332
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.1207199, 26.9715672, -23.6550789, 25.6027508, -50.7234650, 50.6266365
1: -194.0323486, 63.2702599, -186.2450104, 59.6139526, -253.6462708, 249.5152588
2: -104.2892532, 58.2751694, -98.5096741, 55.3716087, -159.6608429, 156.7848358
3: -134.9489288, 46.7267418, -128.8228302, 44.4017982, -179.3507080, 175.5495758
4: -73.6893845, 49.9094925, -69.4679184, 47.3655510, -121.0549316, 119.3774033

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5202022, upper bound: 42.5206964
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5175659, upper bound: 42.5176625
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.1207199, 26.9715672, -23.1128292, 24.9189796, -50.0396957, 50.0843887
1: -194.0323486, 63.2702599, -181.6424103, 58.3029137, -252.3352661, 244.9126434
2: -104.2892532, 58.2751694, -96.5479965, 53.9097023, -158.1989441, 154.8231659
3: -134.9489288, 46.7267418, -125.9141998, 43.2622185, -178.2111511, 172.6409454
4: -73.6893845, 49.9094925, -68.0682220, 46.0008202, -119.6902008, 117.9776993

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5202022, upper bound: 42.5206964
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5175659, upper bound: 42.5176625
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -24.2544250, 26.1291466, -25.0334435, 26.9096889, -51.1641083, 51.1625900
1: -187.9201508, 61.0501823, -192.9767303, 62.9469643, -250.8671112, 254.0269165
2: -100.2535095, 56.5268555, -103.1668091, 58.3489227, -158.6024323, 159.6936493
3: -130.4596405, 45.2624550, -134.0053101, 46.6902618, -177.1499023, 179.2677612
4: -70.9664917, 48.4487038, -73.0678787, 50.0717163, -121.0382080, 121.5165863

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5224412, upper bound: 42.5225512
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222984, upper bound: 42.5222984
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -24.2544250, 26.1291466, -24.2544250, 26.1291466, -50.3835716, 50.3835716
1: -187.9201508, 61.0501823, -187.9201508, 61.0501823, -248.9703369, 248.9703369
2: -100.2535095, 56.5268555, -100.2535095, 56.5268555, -156.7803650, 156.7803650
3: -130.4596405, 45.2624550, -130.4596405, 45.2624550, -175.7220917, 175.7220917
4: -70.9664917, 48.4487038, -70.9664917, 48.4487038, -119.4151917, 119.4151917

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5224412, upper bound: 42.5225512
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222984, upper bound: 42.5222984
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -25.1207199, 26.9715672, -25.0334435, 26.9096889, -52.0303955, 52.0050049
1: -194.0323486, 63.2702599, -192.9767303, 62.9469643, -256.9793091, 256.2469788
2: -104.2892532, 58.2751694, -103.1668091, 58.3489227, -162.6381378, 161.4419556
3: -134.9489288, 46.7267418, -134.0053101, 46.6902618, -181.6391754, 180.7320557
4: -73.6893845, 49.9094925, -73.0678787, 50.0717163, -123.7611008, 122.9773636

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5201051, upper bound: 42.5206828
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5174554, upper bound: 42.5175089
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -25.1207199, 26.9715672, -24.2544250, 26.1291466, -51.2498627, 51.2259865
1: -194.0323486, 63.2702599, -187.9201508, 61.0501823, -255.0825195, 251.1903992
2: -104.2892532, 58.2751694, -100.2535095, 56.5268555, -160.8160858, 158.5286865
3: -134.9489288, 46.7267418, -130.4596405, 45.2624550, -180.2113800, 177.1863861
4: -73.6893845, 49.9094925, -70.9664917, 48.4487038, -122.1380920, 120.8759842

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5201051, upper bound: 42.5206828
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5174554, upper bound: 42.5175089
time: 0.89 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.20 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5241836, upper bound: 42.5242219
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5241560, upper bound: 42.5239917
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5241718, upper bound: 42.5244216
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5241955, upper bound: 42.5241426
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5241946, upper bound: 42.5242003
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5238635
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5241850, upper bound: 42.5244460
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5241207
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5206470, upper bound: 42.5214751
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5199634, upper bound: 42.5201090
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5206470, upper bound: 42.5214751
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5199634, upper bound: 42.5201090
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5206250, upper bound: 42.5213497
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5201857, upper bound: 42.5202201
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5206250, upper bound: 42.5213497
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5201857, upper bound: 42.5202201
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5243707, upper bound: 42.5248637
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5243230, upper bound: 42.5246129
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5254734, upper bound: 42.5255361
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5253425, upper bound: 42.5255361
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5243756, upper bound: 42.5247657
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5242121
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5249431, upper bound: 42.5249454
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5245018
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5206889, upper bound: 42.5215553
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5199898, upper bound: 42.5201819
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5206889, upper bound: 42.5215553
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5199898, upper bound: 42.5201819
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5207980, upper bound: 42.5218156
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5191768, upper bound: 42.5190236
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5207980, upper bound: 42.5218156
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5191768, upper bound: 42.5190236
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5168892, upper bound: 42.5202806
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5193757, upper bound: 42.5206578
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5246219, upper bound: 42.5248602
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5246137, upper bound: 42.5247623
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5246219, upper bound: 42.5248320
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5203329, upper bound: 42.5246137
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5181279, upper bound: 42.5179985
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5217328, upper bound: 42.5227806
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5181279, upper bound: 42.5180203
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5217328, upper bound: 42.5227806
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5167227, upper bound: 42.5159211
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5079494, upper bound: 42.5071085
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5074790, upper bound: 42.5080498
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5033994, upper bound: 42.5035118
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5228079, upper bound: 42.5230474
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5222818, upper bound: 42.5217739
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5226094, upper bound: 42.5230374
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5220130, upper bound: 42.5215269
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5224670, upper bound: 42.5221629
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5217812, upper bound: 42.5217364
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5224670, upper bound: 42.5221832
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5217812, upper bound: 42.5217933
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5204939, upper bound: 42.5213028
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5187680, upper bound: 42.5186794
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5204939, upper bound: 42.5213028
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5187680, upper bound: 42.5186794
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5228517, upper bound: 42.5230702
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5227526, upper bound: 42.5226332
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5228517, upper bound: 42.5230702
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5227526, upper bound: 42.5226332
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5202022, upper bound: 42.5206964
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5175659, upper bound: 42.5176625
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5202022, upper bound: 42.5206964
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5175659, upper bound: 42.5176625
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5224412, upper bound: 42.5225512
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5222984, upper bound: 42.5222984
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5224412, upper bound: 42.5225512
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5222984, upper bound: 42.5222984
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5201051, upper bound: 42.5206828
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5174554, upper bound: 42.5175089
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5201051, upper bound: 42.5206828
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.20
Output dim: 0, lower bound: -42.5174554, upper bound: 42.5175089

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -21.5580864, 23.4122982, -21.3972702, 23.1466465, -44.7047348, 44.8095703
1: -170.8722229, 54.3623390, -168.4158783, 54.0681839, -224.9403992, 222.7782135
2: -90.1412811, 50.5354271, -89.5680618, 49.9290428, -140.0703125, 140.1034393
3: -118.0331726, 40.5885582, -116.8786774, 40.1352196, -158.1683807, 157.4672394
4: -63.3797302, 43.1945343, -63.0987396, 42.6381111, -106.0178299, 106.2932587

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241836, upper bound: 42.5242219
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241836, upper bound: 42.5242219
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -22.1910553, 24.0531826, -21.4612846, 23.2131348, -45.4041901, 45.5144653
1: -174.5082245, 55.9947853, -168.9772644, 54.2463417, -228.7545624, 224.9720459
2: -92.5223236, 51.9097328, -89.8883667, 50.0596581, -142.5819855, 141.7980804
3: -120.8455429, 41.6956406, -117.2751160, 40.2534981, -161.0990448, 158.9707642
4: -65.2748642, 44.4437943, -63.3236809, 42.7474060, -108.0222473, 107.7674713

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241560, upper bound: 42.5239917
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241560, upper bound: 42.5239917
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -22.0459709, 23.9499626, -23.7655048, 25.8438644, -47.8898239, 47.7154694
1: -175.4591064, 55.5393677, -188.2045441, 60.0450706, -235.5041809, 243.7438965
2: -92.3808365, 51.7277527, -99.1468277, 55.8483810, -148.2292023, 150.8745270
3: -120.9885788, 41.4840393, -130.2016296, 44.7937698, -165.7823486, 171.6856689
4: -64.5715179, 44.1949883, -69.9013748, 47.7144737, -112.2859955, 114.0963593

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242310, upper bound: 42.5244216
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242310, upper bound: 42.5244216
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -22.5355530, 24.5289612, -23.7986374, 25.9081039, -48.4436455, 48.3275948
1: -178.7292633, 56.8227921, -188.6399231, 60.1939850, -238.9232330, 245.4626923
2: -94.0935974, 52.9003601, -99.3909302, 55.9698868, -150.0634766, 152.2912903
3: -123.4133224, 42.4434662, -130.5333099, 44.9084930, -168.3218079, 172.9767609
4: -66.1424561, 45.3059082, -70.0891571, 47.8251381, -113.9675903, 115.3950653

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241387, upper bound: 42.5241426
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241387, upper bound: 42.5241426
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -22.2532330, 24.0762787, -21.3972702, 23.1466465, -45.3998718, 45.4735489
1: -174.7256927, 56.0800095, -168.4158783, 54.0681839, -228.7938538, 224.4958801
2: -93.1472626, 51.7382545, -89.5680618, 49.9290428, -143.0762939, 141.3063202
3: -121.1699753, 41.6420441, -116.8786774, 40.1352196, -161.3051605, 158.5207214
4: -65.5276642, 44.2542191, -63.0987396, 42.6381111, -108.1657715, 107.3529434

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241946, upper bound: 42.5242003
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241946, upper bound: 42.5242003
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -22.9347820, 24.7251244, -21.4612846, 23.2131348, -46.1479187, 46.1864090
1: -178.7246552, 57.7095108, -168.9772644, 54.2463417, -232.9709930, 226.6867676
2: -95.5816193, 53.1677971, -89.8883667, 50.0596581, -145.6412811, 143.0561371
3: -124.0870819, 42.7579803, -117.2751160, 40.2534981, -164.3405762, 160.0330963
4: -67.4675903, 45.5207901, -63.3236809, 42.7474060, -110.2149734, 108.8444672

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5238635
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5238635
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -22.2425537, 24.1839886, -23.7655048, 25.8438644, -48.0864029, 47.9494934
1: -175.4053955, 56.1185875, -188.2045441, 60.0450706, -235.4504700, 244.3231354
2: -93.1361847, 51.9014397, -99.1468277, 55.8483810, -148.9845581, 151.0481873
3: -121.4851151, 41.7846375, -130.2016296, 44.7937698, -166.2788849, 171.9862518
4: -65.5193939, 44.4695587, -69.9013748, 47.7144737, -113.2338562, 114.3709259

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241850, upper bound: 42.5244460
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242310, upper bound: 42.5244460
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -22.9285126, 24.8408203, -23.7986374, 25.9081039, -48.8366089, 48.6394501
1: -179.4001617, 57.7529030, -188.6399231, 60.1939850, -239.5941467, 246.3928070
2: -95.5678177, 53.3396225, -99.3909302, 55.9698868, -151.5377045, 152.7305298
3: -124.3959885, 42.9118347, -130.5333099, 44.9084930, -169.3044739, 173.4451447
4: -67.4381180, 45.7438431, -70.0891571, 47.8251381, -115.2632599, 115.8330002

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5240895, upper bound: 42.5241207
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242260, upper bound: 42.5241207
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -20.6611824, 22.3088093, -20.6941319, 22.2622585, -42.9234390, 43.0029411
1: -164.7707520, 52.0913353, -163.5965118, 52.0870590, -216.8577881, 215.6878510
2: -87.1117096, 48.2563972, -86.8954544, 48.1804237, -135.2920990, 135.1518555
3: -113.8181610, 38.7113342, -113.2035904, 38.6436501, -152.4618073, 151.9149170
4: -60.7596169, 41.0032310, -60.7793312, 40.9932594, -101.7528763, 101.7825623

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243230, upper bound: 42.5246129
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243230, upper bound: 42.5246129
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -21.0816669, 22.8431530, -20.6202850, 22.2551861, -43.3368530, 43.4634285
1: -167.6362915, 53.1903191, -163.5702209, 51.9393616, -219.5756531, 216.7605438
2: -88.4503479, 49.3076096, -86.5772018, 48.1006050, -136.5509491, 135.8848114
3: -115.8998642, 39.5737343, -113.1681747, 38.6022758, -154.5021210, 152.7419128
4: -62.0415802, 42.0430336, -60.6008492, 40.9477310, -102.9893112, 102.6438828

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243230, upper bound: 42.5246129
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243230, upper bound: 42.5246129
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -20.7282276, 22.2581577, -21.4916286, 23.1943455, -43.9225693, 43.7497864
1: -163.5390167, 52.2162399, -169.4964905, 54.3453445, -217.8843689, 221.7127380
2: -86.9979935, 48.2105713, -90.2074051, 50.0607033, -137.0587006, 138.4179688
3: -113.3287048, 38.6392212, -117.6583557, 40.2285233, -153.5572205, 156.2975769
4: -60.9785271, 41.0046730, -63.5198326, 42.6538544, -103.6323700, 104.5245056

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5254261, upper bound: 42.5255325
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5254261, upper bound: 42.5255361
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -22.0507755, 23.7933483, -21.9818935, 23.7432232, -45.7939949, 45.7752419
1: -173.5342560, 55.7383041, -173.3091278, 55.5866661, -229.1209259, 229.0474243
2: -92.3993759, 51.4634438, -92.1829376, 51.3123817, -143.7117310, 143.6463776
3: -120.4737167, 41.3166428, -120.2764130, 41.2016335, -161.6753540, 161.5930481
4: -65.0926361, 43.8736572, -64.9225998, 43.7450981, -108.8377380, 108.7962570

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5254863, upper bound: 42.5255325
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5251014, upper bound: 42.5249553
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5243007, upper bound: 42.5249463
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -20.4481926, 22.1768131, -20.6941319, 22.2622585, -42.7104492, 42.8709450
1: -162.8000183, 51.7601738, -163.5965118, 52.0870590, -214.8870544, 215.3566895
2: -86.4000244, 47.6956062, -86.8954544, 48.1804237, -134.5804291, 134.5910645
3: -112.7948151, 38.4254265, -113.2035904, 38.6436501, -151.4384613, 151.6290131
4: -60.5840378, 40.6475830, -60.7793312, 40.9932594, -101.5773010, 101.4269104

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5242121
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5242121
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -21.1095810, 22.8281536, -20.6202850, 22.2551861, -43.3647652, 43.4484406
1: -166.7941437, 53.3734818, -163.5702209, 51.9393616, -218.7334900, 216.9436951
2: -88.7927246, 49.1233940, -86.5772018, 48.1006050, -136.8933258, 135.7005920
3: -115.6873856, 39.5410919, -113.1681747, 38.6022758, -154.2896576, 152.7092438
4: -62.4593277, 41.9053421, -60.6008492, 40.9477310, -103.4070587, 102.5061951

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5242121
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242228, upper bound: 42.5242121
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -21.8344688, 23.6307068, -21.6899185, 23.4590664, -45.2935333, 45.3206253
1: -172.5440826, 55.1696053, -171.0946503, 54.8478203, -227.3919067, 226.2642517
2: -91.9907379, 50.7999878, -90.9435196, 50.6732826, -142.6640167, 141.7434998
3: -119.6751785, 40.9005165, -118.7123337, 40.6920509, -160.3672333, 159.6128387
4: -64.5853882, 43.3307152, -64.0569992, 43.2253036, -107.8106918, 107.3877106

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249308, upper bound: 42.5249244
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5249308, upper bound: 42.5249454
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -22.4421177, 24.2238064, -21.7605419, 23.5291634, -45.9712677, 45.9843483
1: -176.1313934, 56.6551781, -171.6355896, 55.0281219, -231.1595001, 228.2907715
2: -94.1679382, 52.0987129, -91.2461853, 50.8110085, -144.9789124, 143.3448944
3: -122.2881165, 41.9196930, -119.0946732, 40.8095512, -163.0976715, 161.0143280
4: -66.3065186, 44.4940262, -64.2762375, 43.3409500, -109.6474686, 108.7702560

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247820, upper bound: 42.5245018
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5247820, upper bound: 42.5245018
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -22.2751045, 24.0984688, -24.6576233, 26.5210495, -48.7961540, 48.7560921
1: -175.8851318, 56.2734261, -190.1853333, 61.9312057, -237.8162994, 246.4587555
2: -93.7635651, 51.8145714, -101.5482941, 57.4629898, -151.2265625, 153.3628693
3: -122.0016251, 41.7147598, -131.9462433, 45.9770508, -167.9786682, 173.6610107
4: -65.8517685, 44.2004433, -71.9221725, 49.3335991, -115.1853638, 116.1226120

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5230244, upper bound: 42.5232921
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5223020, upper bound: 42.5232838
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -22.2751045, 24.0984688, -23.8876629, 25.7384052, -48.0135117, 47.9861298
1: -175.8851318, 56.2734261, -185.0470581, 60.0589409, -235.9440765, 241.3204803
2: -93.7635651, 51.8145714, -98.6618958, 55.6578712, -149.4214325, 150.4764404
3: -122.0016251, 41.7147598, -128.3914490, 44.5622635, -166.5638428, 170.1062012
4: -65.8517685, 44.2004433, -69.8271790, 47.7203026, -113.5720673, 114.0276184

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5196864, upper bound: 42.5195941
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5184873, upper bound: 42.5189376
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23.8534966, 25.9221058, -23.8725815, 25.9058189, -49.7593155, 49.7946854
1: -188.9317780, 60.3911209, -188.3182678, 60.4276390, -249.3593903, 248.7093811
2: -100.3371201, 55.8300743, -100.0406036, 55.8600807, -156.1972046, 155.8706818
3: -130.9298706, 44.8159180, -130.5545044, 44.8262405, -175.7561035, 175.3704224
4: -70.6357498, 47.6533623, -70.5501633, 47.6903572, -118.3260956, 118.2035217

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241644, upper bound: 42.5240816
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238111, upper bound: 42.5239540
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -23.3093567, 25.3168488, -24.3192482, 26.4087296, -49.7180824, 49.6360970
1: -183.7537384, 58.9758949, -191.9681244, 61.5726967, -245.3264313, 250.9440002
2: -97.7682800, 54.5277481, -101.8611450, 56.9668007, -154.7350769, 156.3888702
3: -127.4606934, 43.7520790, -133.0699310, 45.7001953, -173.1608887, 176.8219910
4: -68.9196548, 46.6138153, -71.8292694, 48.6536751, -117.5733261, 118.4430847

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5241259, upper bound: 42.5240042
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5237786, upper bound: 42.5238973
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -24.3403187, 26.3856106, -23.4734745, 25.4708061, -49.8111115, 49.8590736
1: -190.4874573, 61.4968414, -184.9656830, 59.4093170, -249.8967438, 246.4625092
2: -101.6388016, 56.7457695, -98.3393784, 54.8837433, -156.5225220, 155.0851135
3: -132.1451111, 45.5913696, -128.2363892, 44.0636063, -176.2086639, 173.8277588
4: -71.8850174, 48.6349983, -69.3910675, 46.8745995, -118.7595901, 118.0260544

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5242008, upper bound: 42.5240736
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5238380, upper bound: 42.5239506
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -23.8728695, 25.8712482, -23.9277325, 25.9803982, -49.8532639, 49.7989807
1: -185.8412476, 60.2590866, -188.6583862, 60.5729179, -246.4141541, 248.9174500
2: -99.3455276, 55.6187744, -100.1912155, 56.0062485, -155.3517609, 155.8099976
3: -129.0326996, 44.6646538, -130.7870636, 44.9499283, -173.9826355, 175.4516907
4: -70.3763123, 47.7597122, -70.6922455, 47.8522072, -118.2285080, 118.4519501

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217130, upper bound: 42.5222660
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5244481, upper bound: 42.5244481
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -24.2693825, 26.2453365, -24.7044525, 26.4689980, -50.7383804, 50.9497910
1: -190.2330933, 61.3139496, -190.5687408, 62.2247581, -252.4578400, 251.8826904
2: -101.5622635, 56.6057014, -102.6396713, 57.2202148, -158.7824707, 159.2453766
3: -132.0807800, 45.3993225, -132.6580658, 45.8635521, -177.9443359, 178.0573883
4: -71.6198883, 48.3823280, -72.5263748, 48.9553299, -120.5752182, 120.9086838

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5181279, upper bound: 42.5232404
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5181279, upper bound: 42.5232404
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -24.2693825, 26.2453365, -25.4913998, 27.4659500, -51.7353325, 51.7367363
1: -190.2330933, 61.3139496, -197.2038879, 64.2941818, -254.5272675, 258.5178223
2: -101.5622635, 56.6057014, -105.7585144, 59.3844604, -160.9467010, 162.3642120
3: -132.0807800, 45.3993225, -137.1338043, 47.5339699, -179.6147461, 182.5331116
4: -71.6198883, 48.3823280, -74.8460541, 50.9084549, -122.5283432, 123.2283707

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217328, upper bound: 42.5227806
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217328, upper bound: 42.5227806
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -23.9096889, 25.7147274, -21.3746796, 22.9141655, -46.8238525, 47.0894089
1: -184.9740601, 60.1828728, -167.9140625, 53.7813301, -238.7553864, 228.0969238
2: -99.2997360, 55.4925308, -89.6732407, 49.6140289, -148.9137421, 145.1657715
3: -128.5708008, 44.5075645, -116.4226303, 39.7396355, -168.3104248, 160.9301910
4: -70.0903625, 47.5466118, -62.8630257, 42.2229500, -112.3133011, 110.4096298

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228079, upper bound: 42.5230474
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5228079, upper bound: 42.5230474
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -24.5395546, 26.3332424, -21.3077602, 22.8830624, -47.4226151, 47.6410027
1: -188.8793945, 61.7324409, -167.4046021, 53.6058540, -242.4852295, 229.1370392
2: -101.6248245, 56.8440018, -89.3069229, 49.5187263, -151.1435394, 146.1509247
3: -131.3929291, 45.5739403, -116.0217743, 39.6729927, -171.0659027, 161.5957031
4: -71.8968430, 48.7477264, -62.6495667, 42.1872749, -114.0841141, 111.3972931

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222818, upper bound: 42.5217739
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222818, upper bound: 42.5217739
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -24.3626709, 26.2322388, -22.2679615, 24.0794601, -48.4421310, 48.5001984
1: -188.6999969, 61.3344460, -175.7713470, 56.1327209, -244.8327026, 237.1057892
2: -101.1700439, 56.6017990, -92.9737701, 52.0506134, -153.2206573, 149.5755615
3: -131.1024780, 45.4104691, -121.6139908, 41.7644958, -172.8669434, 167.0244141
4: -71.4000168, 48.5143661, -65.5134354, 44.4364891, -115.8365021, 114.0278015

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5226094, upper bound: 42.5230374
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5226094, upper bound: 42.5230374
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -25.0209618, 26.8786030, -22.2836781, 24.1198444, -49.1408081, 49.1622810
1: -192.7646484, 62.9538002, -176.0419159, 56.2474632, -249.0121155, 238.9956970
2: -103.5980988, 58.0182076, -93.1658401, 52.1389160, -155.7369843, 151.1840515
3: -134.0406342, 46.5129890, -121.8126450, 41.8517952, -175.8924255, 168.3256378
4: -73.2850952, 49.7757263, -65.6684647, 44.5162888, -117.8013687, 115.4441757

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5220130, upper bound: 42.5215269
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5220130, upper bound: 42.5215269
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -24.9055557, 26.7666683, -25.0334702, 26.9097290, -51.8152847, 51.8001404
1: -192.0485687, 62.6275177, -192.9769745, 62.9470634, -254.9956207, 255.6044464
2: -102.6625443, 58.0459938, -103.1669617, 58.3489990, -161.0115204, 161.2129517
3: -133.3527069, 46.4457703, -134.0055237, 46.6903191, -180.0430298, 180.4512634
4: -72.7086182, 49.7978134, -73.0679855, 50.0717926, -122.7804031, 122.8657990

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217812, upper bound: 42.5217364
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217812, upper bound: 42.5217364
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -24.8945751, 26.6957722, -24.4724331, 26.3157272, -51.2102966, 51.1682053
1: -193.0835266, 62.4991760, -188.6270752, 61.5911369, -254.6746674, 251.1262512
2: -103.1610870, 57.9626236, -101.0001450, 57.0380287, -160.1991119, 158.9627686
3: -133.8247375, 46.3036995, -131.1353455, 45.6537476, -179.4784698, 177.4390411
4: -72.4889679, 49.5898018, -71.4984589, 48.9311600, -121.4201279, 121.0882492

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5077560, upper bound: 42.5066782
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5026591, upper bound: 42.5026591
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -24.9055557, 26.7666683, -24.2544250, 26.1291466, -51.0347023, 51.0210915
1: -192.0485687, 62.6275177, -187.9201508, 61.0501823, -253.0987549, 250.5476685
2: -102.6625443, 58.0459938, -100.2535095, 56.5268555, -159.1893921, 158.2994995
3: -133.3527069, 46.4457703, -130.4596405, 45.2624550, -178.6151581, 176.9054108
4: -72.7086182, 49.7978134, -70.9664917, 48.4487038, -121.1573181, 120.7643051

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5222385, upper bound: 42.5219475
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5221824, upper bound: 42.5219422
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -24.8945751, 26.6957722, -23.6940308, 25.5368862, -50.4314613, 50.3898010
1: -193.0835266, 62.4991760, -183.7513580, 59.6881104, -252.7716064, 246.2505341
2: -103.1610870, 57.9626236, -98.0897675, 55.2002373, -158.3613281, 156.0523987
3: -133.8247375, 46.3036995, -127.5832748, 44.2271652, -178.0518799, 173.8869781
4: -72.4889679, 49.5898018, -69.4056091, 47.3039284, -119.7928925, 118.9953995

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5216097, upper bound: 42.5217251
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5216749, upper bound: 42.5217169
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23.1075497, 24.9155750, -23.2942276, 25.2281837, -48.3357315, 48.2097969
1: -179.8965302, 58.1174965, -183.5260773, 58.6559067, -238.5524292, 241.6435394
2: -95.7892990, 53.9035759, -96.9263306, 54.5169983, -150.3063049, 150.8299103
3: -124.6250381, 43.1625977, -126.8929825, 43.7156410, -168.3406372, 170.0555725
4: -67.6151428, 46.1540680, -68.3418427, 46.6500473, -114.2651825, 114.4959106

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5182474, upper bound: 42.5183166
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5182605, upper bound: 42.5183173
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -23.5542774, 25.4375343, -23.3167057, 25.2641582, -48.8184357, 48.7542419
1: -182.8112793, 59.3423538, -183.7517395, 58.7686386, -241.5799103, 243.0940857
2: -97.4494629, 54.9625664, -97.1253662, 54.5972061, -152.0466614, 152.0879364
3: -126.8619385, 44.0399895, -127.0693741, 43.7958717, -170.6578064, 171.1093597
4: -69.0029144, 47.1499863, -68.5017853, 46.7225380, -115.7254486, 115.6517410

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5182324, upper bound: 42.5183145
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5182155, upper bound: 42.5182894
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23.1075497, 24.9155750, -22.7532444, 24.5397930, -47.6473427, 47.6688156
1: -179.8965302, 58.1174965, -178.8826447, 57.3221703, -237.2187042, 237.0001221
2: -95.7892990, 53.9035759, -94.9333725, 53.0503578, -148.8396301, 148.8369446
3: -124.6250381, 43.1625977, -123.8966980, 42.5709419, -167.1959839, 167.0592957
4: -67.6151428, 46.1540680, -66.9381714, 45.2816811, -112.8968201, 113.0922394

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5182108, upper bound: 42.5182489
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5182109, upper bound: 42.5182489
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -23.5542774, 25.4375343, -22.7750683, 24.5830879, -48.1373672, 48.2126007
1: -182.8112793, 59.3423538, -179.1887207, 57.4613342, -240.2725830, 238.5310669
2: -97.4494629, 54.9625664, -95.1684723, 53.1430092, -150.5924683, 150.1310425
3: -126.8619385, 44.0399895, -124.1700439, 42.6640320, -169.5259705, 168.2100372
4: -69.0029144, 47.1499863, -67.1069412, 45.3660088, -114.3689270, 114.2568970

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5181800, upper bound: 42.5182603
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -42.5181800, upper bound: 42.5182454
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23.1075497, 24.9155750, -24.6576233, 26.5210495, -49.6286011, 49.5731926
1: -179.8965302, 58.1174965, -190.1853333, 61.9312057, -241.8277130, 248.3028259
2: -95.7892990, 53.9035759, -101.5482941, 57.4629898, -153.2522583, 155.4518738
3: -124.6250381, 43.1625977, -131.9462433, 45.9770508, -170.6020813, 175.1088409
4: -67.6151428, 46.1540680, -71.9221725, 49.3335991, -116.9487305, 118.0762405

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5219475, upper bound: 42.5221824
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217188, upper bound: 42.5215508
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -23.5542774, 25.4375343, -24.6957493, 26.5809212, -50.1351967, 50.1332855
1: -182.8112793, 59.3423538, -190.5820007, 62.1321144, -244.9433746, 249.9243469
2: -97.4494629, 54.9625664, -101.8358154, 57.6014671, -155.0509338, 156.7983856
3: -126.8619385, 44.0399895, -132.3112183, 46.1066284, -172.9685364, 176.3512115
4: -69.0029144, 47.1499863, -72.1379166, 49.4509163, -118.4538269, 119.2878799

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5219422, upper bound: 42.5223021
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -42.5217169, upper bound: 42.5215834
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -23.1075497, 24.9155750, -23.8876629, 25.7384052, -48.8459549, 48.8032379
1: -179.8965302, 58.1174965, -185.0470581, 60.0589409, -239.9554749, 243.1645508
2: -95.7892990, 53.9035759, -98.6618958, 55.6578712, -151.4471588, 152.5654602
3: -124.6250381, 43.1625977, -128.3914490, 44.5622635, -169.1872559, 171.5540466
4: -67.6151428, 46.1540680, -69.8271790, 47.7203026, -115.3354263, 115.9812469

Time for backsubstitution: 1.38 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.43 + 417.21 = 420.64 seconds
