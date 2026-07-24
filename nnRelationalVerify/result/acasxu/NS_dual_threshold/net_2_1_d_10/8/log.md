## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 77.93799558274


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173)
1: (-32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176)
2: (-28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595)
3: (-39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838)
4: (-36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 2.21 = 4.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -77.9535863, upper bound: 77.9535863

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525704, upper bound: 77.9529948
time: 0.69 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525704, upper bound: 77.9526590
time: 1.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.88 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 3, lower bound: -77.9525704, upper bound: 77.9529948
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 3, lower bound: -77.9525704, upper bound: 77.9526590

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -41.9465866, 50.7789307, -78.9649811, 75.4661636
1: -21.7694626, 26.5227585, -32.5055008, 40.2750320, -62.0444946, 59.0282516
2: -18.9085445, 26.6295929, -28.2902870, 40.3181725, -59.2267151, 54.9198723
3: -25.9847813, 31.7821980, -39.0206070, 48.2061768, -74.1909561, 70.8028030
4: -24.4721470, 35.5742149, -36.7614212, 53.8471222, -78.3192673, 72.3356094

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9327010, upper bound: 77.9525126
time: 0.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525624, upper bound: 77.9529898
time: 0.76 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -41.9465866, 50.7789307, -92.2251587, 92.1385880
1: -32.1180115, 39.8035698, -32.5055008, 40.2750320, -72.3930435, 72.3090668
2: -27.9553337, 39.8533745, -28.2902870, 40.3181725, -68.2735062, 68.1436615
3: -38.5680542, 47.6387863, -39.0206070, 48.2061768, -86.7742310, 86.6593933
4: -36.3231392, 53.2251053, -36.7614212, 53.8471222, -90.1702576, 89.9865112

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9522995, upper bound: 77.9459729
time: 0.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9459570
time: 1.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.54 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -77.9327010, upper bound: 77.9525126
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -77.9525624, upper bound: 77.9529898
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -77.9522995, upper bound: 77.9459729
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9459570

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -38.9059372, 47.1887283, -75.3747711, 72.4255142
1: -21.7694626, 26.5227585, -30.2153473, 37.4228210, -59.1922836, 56.7381020
2: -18.9085445, 26.6295929, -26.3050613, 37.4508362, -56.3593826, 52.9346542
3: -25.9847813, 31.7821980, -36.3253708, 44.7629814, -70.7477646, 68.1075668
4: -24.4721470, 35.5742149, -34.1707458, 50.0052338, -74.4773788, 69.7449417

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9256675, upper bound: 77.9522372
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9253246, upper bound: 77.9395630
time: 1.03 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -41.6742859, 50.4779816, -78.6640320, 75.1938553
1: -21.7694626, 26.5227585, -32.2958031, 40.0290489, -61.7985077, 58.8185616
2: -18.9085445, 26.6295929, -28.1075344, 40.0769844, -58.9855270, 54.7371254
3: -25.9847813, 31.7821980, -38.7702141, 47.9131126, -73.8978958, 70.5524063
4: -24.4721470, 35.5742149, -36.5272942, 53.5238190, -77.9959641, 72.1015091

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525496, upper bound: 77.9510162
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9516277, upper bound: 77.9510550
time: 1.52 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -39.7767181, 47.9650536, -89.4112854, 89.9687042
1: -32.1180115, 39.8035698, -30.7269535, 38.0118408, -70.1298523, 70.5305252
2: -27.9553337, 39.8533745, -26.7471428, 38.0404625, -65.9957886, 66.6005096
3: -38.5680542, 47.6387863, -36.9256668, 45.4790421, -84.0470963, 84.5644531
4: -36.3231392, 53.2251053, -34.7300873, 50.7894363, -87.1125717, 87.9551926

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9503100, upper bound: 77.9282770
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9524246, upper bound: 77.9459499
time: 0.81 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -43.1615906, 52.6771889, -94.1234131, 93.3535919
1: -32.1180115, 39.8035698, -33.5264969, 41.7147179, -73.8327332, 73.3300476
2: -27.9553337, 39.8533745, -29.1779251, 41.8765984, -69.8319321, 69.0312958
3: -38.5680542, 47.6387863, -40.4070930, 49.8904648, -88.4585190, 88.0458755
4: -36.3231392, 53.2251053, -37.9242630, 55.9812622, -92.3043976, 91.1493683

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250
time: 1.11 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.82 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9256675, upper bound: 77.9522372
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9253246, upper bound: 77.9395630
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9525496, upper bound: 77.9510162
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9516277, upper bound: 77.9510550
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9503100, upper bound: 77.9282770
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9524246, upper bound: 77.9459499
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -24.1045227, 28.3574142, -38.9059372, 47.1887283, -71.2932510, 67.2633514
1: -18.5040874, 22.3795414, -30.2153473, 37.4228210, -55.9268990, 52.5948868
2: -16.0840149, 22.4593487, -26.3050613, 37.4508362, -53.5348511, 48.7644119
3: -22.1233826, 26.7815647, -36.3253708, 44.7629814, -66.8863678, 63.1069260
4: -20.7617455, 29.9629669, -34.1707458, 50.0052338, -70.7669754, 64.1336899

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9221022, upper bound: 77.9517444
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9217826, upper bound: 77.9401222
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -34.8093834, 41.3694382, -38.9059372, 47.1887283, -81.9981079, 80.2753754
1: -26.7821999, 32.6442871, -30.2153473, 37.4228210, -64.2050171, 62.8596344
2: -23.3394756, 32.7738380, -26.3050613, 37.4508362, -60.7903061, 59.0788956
3: -32.3881607, 38.9999428, -36.3253708, 44.7629814, -77.1511383, 75.3253021
4: -30.1711388, 43.8623695, -34.1707458, 50.0052338, -80.1763611, 78.0330963

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9226539, upper bound: 77.9392403
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9253246, upper bound: 77.9395630
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -40.2415161, 48.2965202, -76.4825516, 73.7610931
1: -21.7694626, 26.5227585, -31.0388165, 38.2832260, -60.0526886, 57.5615692
2: -18.9085445, 26.6295929, -27.0241070, 38.2253265, -57.1338730, 53.6536942
3: -25.9847813, 31.7821980, -37.2061462, 45.8390923, -71.8238678, 68.9883423
4: -24.4721470, 35.5742149, -35.0574188, 51.0426140, -75.5147400, 70.6316147

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9524127, upper bound: 77.9503104
time: 1.07 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525086, upper bound: 77.9510079
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -41.1443329, 49.7956924, -77.9817352, 74.6638947
1: -21.7694626, 26.5227585, -31.8737583, 39.4841576, -61.2536163, 58.3965149
2: -18.9085445, 26.6295929, -27.7399025, 39.5278969, -58.4364395, 54.3694954
3: -25.9847813, 31.7821980, -38.2511063, 47.2646751, -73.2494431, 70.0333023
4: -24.4721470, 35.5742149, -36.0435677, 52.7837029, -77.2558517, 71.6177673

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393755, upper bound: 77.9505726
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9390453, upper bound: 77.9399125
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -38.4355240, 46.6413689, -39.7767181, 47.9650536, -86.4005737, 86.4180756
1: -29.8471088, 36.9797592, -30.7269535, 38.0118408, -67.8589478, 67.7067108
2: -25.9868698, 37.0171318, -26.7471428, 38.0404625, -64.0273209, 63.7642593
3: -35.8958549, 44.2278976, -36.9256668, 45.4790421, -81.3748932, 81.1535568
4: -33.7544556, 49.4227867, -34.7300873, 50.7894363, -84.5438919, 84.1528778

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9333459, upper bound: 77.9274824
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9333091, upper bound: 77.9248999
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.1717911, 49.8883514, -39.7767181, 47.9650536, -89.1368408, 89.6650467
1: -31.9071941, 39.5558090, -30.7269535, 38.0118408, -69.9190369, 70.2827530
2: -27.7717896, 39.6101379, -26.7471428, 38.0404625, -65.8122406, 66.3572769
3: -38.3161316, 47.3437538, -36.9256668, 45.4790421, -83.7951584, 84.2694092
4: -36.0879478, 52.8990135, -34.7300873, 50.7894363, -86.8773804, 87.6290894

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9338058, upper bound: 77.9451553
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -30.2205067, 36.2791977, -77.7254257, 80.4125061
1: -32.1180115, 39.8035698, -23.3760128, 28.6446552, -60.7626648, 63.1795731
2: -27.9553337, 39.8533745, -20.3243904, 28.8434830, -56.7988167, 60.1777649
3: -38.5680542, 47.6387863, -28.0764751, 34.2764206, -72.8444748, 75.7152634
4: -36.3231392, 53.2251053, -26.3163967, 38.5819016, -74.9050446, 79.5414886

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9438194, upper bound: 77.9281277
time: 1.08 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458005
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -42.6665916, 52.1030006, -93.5492249, 92.8585968
1: -32.1180115, 39.8035698, -33.1441917, 41.2458954, -73.3639069, 72.9477615
2: -27.9553337, 39.8533745, -28.8452950, 41.4192886, -69.3746185, 68.6986694
3: -38.5680542, 47.6387863, -39.9617653, 49.3260345, -87.8940887, 87.6005554
4: -36.3231392, 53.2251053, -37.4890594, 55.3692169, -91.6923523, 90.7141647

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9089481, upper bound: 77.9438194
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458006
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.51 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9221022, upper bound: 77.9517444
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9217826, upper bound: 77.9401222
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9226539, upper bound: 77.9392403
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9253246, upper bound: 77.9395630
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9524127, upper bound: 77.9503104
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9525086, upper bound: 77.9510079
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9393755, upper bound: 77.9505726
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9390453, upper bound: 77.9399125
NS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9333459, upper bound: 77.9274824
NS_A2_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9333091, upper bound: 77.9248999
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9338058, upper bound: 77.9451553
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9438194, upper bound: 77.9281277
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458005
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9089481, upper bound: 77.9438194
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458006

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -22.9702396, 26.8798943, -38.9059372, 47.1887283, -70.1589584, 65.7858276
1: -17.5727367, 21.1764984, -30.2153473, 37.4228210, -54.9955559, 51.3918457
2: -15.2716131, 21.2516975, -26.3050613, 37.4508362, -52.7224503, 47.5567589
3: -21.0244789, 25.3388939, -36.3253708, 44.7629814, -65.7874603, 61.6642647
4: -19.7019234, 28.3351135, -34.1707458, 50.0052338, -69.7071533, 62.5058479

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9212644, upper bound: 77.9515701
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9221022, upper bound: 77.9516451
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -38.9059372, 47.1887283, -73.3004456, 69.3657684
1: -19.9750519, 24.0622005, -30.2153473, 37.4228210, -57.3978729, 54.2775497
2: -17.3923473, 24.0895405, -26.3050613, 37.4508362, -54.8431816, 50.3945999
3: -23.9274788, 28.7653179, -36.3253708, 44.7629814, -68.6904602, 65.0906906
4: -22.4313278, 32.1443291, -34.1707458, 50.0052338, -72.4365463, 66.3150558

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.8507218, upper bound: 77.9196702
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.8488643, upper bound: 77.9235389
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -34.8093834, 41.3694382, -36.8556671, 44.4881668, -79.2975388, 78.2250977
1: -26.7821999, 32.6442871, -28.5131187, 35.2498283, -62.0320282, 61.1574059
2: -23.3394756, 32.7738380, -24.8311615, 35.2558250, -58.5952911, 57.6049957
3: -32.3881607, 38.9999428, -34.3198280, 42.1420364, -74.5301743, 73.3197708
4: -30.1711388, 43.8623695, -32.2276268, 47.0565186, -77.2276535, 76.0899963

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9131754, upper bound: 77.9368720
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9217970, upper bound: 77.9379248
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -34.8093834, 41.3694382, -40.2369270, 49.1826782, -83.9920654, 81.6063690
1: -26.7821999, 32.6442871, -31.3004303, 38.9267883, -65.7089844, 63.9447174
2: -23.3394756, 32.7738380, -27.2503815, 39.0684662, -62.4079323, 60.0242119
3: -32.3881607, 38.9999428, -37.7908020, 46.5242157, -78.9123688, 76.7907410
4: -30.1711388, 43.8623695, -35.4032402, 52.2151566, -82.3862762, 79.2655869

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9217959, upper bound: 77.9387878
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9217822, upper bound: 77.9392343
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -37.3881073, 44.3226242, -72.5086670, 70.9076691
1: -21.7694626, 26.5227585, -28.6501770, 35.0685577, -56.8380203, 55.1729240
2: -18.9085445, 26.6295929, -24.9549236, 34.9062653, -53.8148079, 51.5845070
3: -25.9847813, 31.7821980, -34.3252068, 41.9741554, -67.9589386, 66.1073990
4: -24.4721470, 35.5742149, -32.3147926, 46.5780640, -71.0502090, 67.8889999

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9506264, upper bound: 77.9489370
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9517012, upper bound: 77.9493355
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -40.2098045, 48.2561150, -76.4421616, 73.7293854
1: -21.7694626, 26.5227585, -31.0128021, 38.2506714, -60.0201340, 57.5355568
2: -18.9085445, 26.6295929, -27.0015354, 38.1926994, -57.1012421, 53.6311264
3: -25.9847813, 31.7821980, -37.1748390, 45.8001518, -71.7849274, 68.9570389
4: -24.4721470, 35.5742149, -35.0280151, 50.9986496, -75.4707947, 70.6022339

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9507499, upper bound: 77.9494975
time: 1.22 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9517675, upper bound: 77.9500330
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -24.1045227, 28.3574142, -41.1443329, 49.7956924, -73.9002151, 69.5017395
1: -18.5040874, 22.3795414, -31.8737583, 39.4841576, -57.9882393, 54.2532997
2: -16.0840149, 22.4593487, -27.7399025, 39.5278969, -55.6119118, 50.1992493
3: -22.1233826, 26.7815647, -38.2511063, 47.2646751, -69.3880386, 65.0326691
4: -20.7617455, 29.9629669, -36.0435677, 52.7837029, -73.5454483, 66.0065231

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387837, upper bound: 77.9499795
time: 0.91 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384725, upper bound: 77.9405978
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -34.8093834, 41.3694382, -41.1443329, 49.7956924, -84.6050720, 82.5137634
1: -26.7821999, 32.6442871, -31.8737583, 39.4841576, -66.2663574, 64.5180435
2: -23.3394756, 32.7738380, -27.7399025, 39.5278969, -62.8673630, 60.5137405
3: -32.3881607, 38.9999428, -38.2511063, 47.2646751, -79.6528091, 77.2510376
4: -30.1711388, 43.8623695, -36.0435677, 52.7837029, -82.9548416, 79.9059296

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9334518, upper bound: 77.9392810
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9390453, upper bound: 77.9399125
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -36.1700401, 43.4971924, -39.7767181, 47.9650536, -84.1350937, 83.2739029
1: -27.9390888, 34.4819832, -30.7269535, 38.0118408, -65.9509277, 65.2089233
2: -24.3269978, 34.4814186, -26.7471428, 38.0404625, -62.3674469, 61.2285576
3: -33.5344810, 41.2522278, -36.9256668, 45.4790421, -79.0135193, 78.1778946
4: -31.5612526, 46.0148773, -34.7300873, 50.7894363, -82.3506927, 80.7449646

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -48.3010254, 58.6476707, -39.7767181, 47.9650536, -96.2660828, 98.4243927
1: -37.4080391, 46.4722481, -30.7269535, 38.0118408, -75.4198761, 77.1992035
2: -32.6320572, 46.5963326, -26.7471428, 38.0404625, -70.6725006, 73.3434601
3: -45.2977142, 55.4843941, -36.9256668, 45.4790421, -90.7767563, 92.4100571
4: -42.3785172, 62.3377914, -34.7300873, 50.7894363, -93.1679459, 97.0678787

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -38.4355240, 46.6413689, -30.2205067, 36.2791977, -74.7147217, 76.8618774
1: -29.8471088, 36.9797592, -23.3760128, 28.6446552, -58.4917641, 60.3557663
2: -25.9868698, 37.0171318, -20.3243904, 28.8434830, -54.8303528, 57.3415222
3: -35.8958549, 44.2278976, -28.0764751, 34.2764206, -70.1722565, 72.3043671
4: -33.7544556, 49.4227867, -26.3163967, 38.5819016, -72.3363571, 75.7391739

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9467073, upper bound: 77.9255451
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395630, upper bound: 77.9253246
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -41.1717911, 49.8883514, -30.2205067, 36.2791977, -77.4509888, 80.1088486
1: -31.9071941, 39.5558090, -23.3760128, 28.6446552, -60.5518494, 62.9318085
2: -27.7717896, 39.6101379, -20.3243904, 28.8434830, -56.6152725, 59.9345284
3: -38.3161316, 47.3437538, -28.0764751, 34.2764206, -72.5925446, 75.4202271
4: -36.0879478, 52.8990135, -26.3163967, 38.5819016, -74.6698456, 79.2153702

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9470569, upper bound: 77.9392658
time: 1.11 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399125, upper bound: 77.9390453
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -39.7738800, 48.6438789, -90.0901031, 89.9658813
1: -32.1180115, 39.8035698, -30.9381809, 38.4866905, -70.6047058, 70.7417374
2: -27.9553337, 39.8533745, -26.9366169, 38.6396027, -66.5949402, 66.7899780
3: -38.5680542, 47.6387863, -37.3691788, 45.9937096, -84.5617523, 85.0079651
4: -36.3231392, 53.2251053, -34.9922638, 51.6398849, -87.9630280, 88.2173615

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9253600, upper bound: 77.9438194
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9252975, upper bound: 77.9376606
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -41.4462280, 50.1920013, -42.4002914, 51.8007088, -93.2469330, 92.5922928
1: -32.1180115, 39.8035698, -32.9388084, 41.0035210, -73.1215363, 72.7423782
2: -27.9553337, 39.8533745, -28.6668015, 41.1806908, -69.1360245, 68.5201721
3: -38.5680542, 47.6387863, -39.7164993, 49.0372391, -87.6052933, 87.3552856
4: -36.3231392, 53.2251053, -37.2602005, 55.0491829, -91.3723068, 90.4853058

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387814, upper bound: 77.9453331
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.66 seconds
NS_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9212644, upper bound: 77.9515701
NS_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9221022, upper bound: 77.9516451
NS_A1_B1_A1_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.8507218, upper bound: 77.9196702
NS_A1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.8488643, upper bound: 77.9235389
NS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9131754, upper bound: 77.9368720
NS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9217970, upper bound: 77.9379248
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9217959, upper bound: 77.9387878
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9217822, upper bound: 77.9392343
NS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9506264, upper bound: 77.9489370
NS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9517012, upper bound: 77.9493355
NS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9507499, upper bound: 77.9494975
NS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9517675, upper bound: 77.9500330
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9387837, upper bound: 77.9499795
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9384725, upper bound: 77.9405978
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9334518, upper bound: 77.9392810
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9390453, upper bound: 77.9399125
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9336587, upper bound: 77.9386206
NS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9467073, upper bound: 77.9255451
NS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9395630, upper bound: 77.9253246
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9470569, upper bound: 77.9392658
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9399125, upper bound: 77.9390453
NS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9253600, upper bound: 77.9438194
NS_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9252975, upper bound: 77.9376606
NS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9387814, upper bound: 77.9453331
NS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588

## BFS NS instance: NS_A1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -21.6248856, 24.6825733, -38.9059372, 47.1887283, -68.8136139, 63.5885086
1: -16.3540230, 19.4223385, -30.2153473, 37.4228210, -53.7768364, 49.6376877
2: -14.2361240, 19.4072628, -26.3050613, 37.4508362, -51.6869583, 45.7123260
3: -19.5663929, 23.2162151, -36.3253708, 44.7629814, -64.3293762, 59.5415878
4: -18.3040810, 25.8403549, -34.1707458, 50.0052338, -68.3092957, 60.0110970

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9034696, upper bound: 77.9502207
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9034696, upper bound: 77.9515701
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -22.8986530, 26.7912273, -38.9059372, 47.1887283, -70.0873795, 65.6971588
1: -17.5167942, 21.1046333, -30.2153473, 37.4228210, -54.9396133, 51.3199806
2: -15.2222376, 21.1801529, -26.3050613, 37.4508362, -52.6730728, 47.4852142
3: -20.9571171, 25.2526760, -36.3253708, 44.7629814, -65.7200928, 61.5780487
4: -19.6372547, 28.2393932, -34.1707458, 50.0052338, -69.6424789, 62.4101295

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9062387, upper bound: 77.9499455
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9212100, upper bound: 77.9509991
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -33.4673347, 39.6504440, -40.2369270, 49.1826782, -82.6500092, 79.8873749
1: -25.7021122, 31.2593193, -31.3004303, 38.9267883, -64.6288986, 62.5597496
2: -22.3965511, 31.3766556, -27.2503815, 39.0684662, -61.4650192, 58.6270294
3: -31.1130371, 37.3390770, -37.7908020, 46.5242157, -77.6372528, 75.1298828
4: -28.9364624, 41.9842148, -35.4032402, 52.2151566, -81.1516113, 77.3874283

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9067702, upper bound: 77.9365357
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9209037, upper bound: 77.9375893
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -35.3306732, 41.7020187, -40.2369270, 49.1826782, -84.5133514, 81.9389496
1: -27.1119175, 32.9430580, -31.3004303, 38.9267883, -66.0387039, 64.2434845
2: -23.6637344, 33.0150185, -27.2503815, 39.0684662, -62.7321892, 60.2653999
3: -32.8386917, 39.3118210, -37.7908020, 46.5242157, -79.3629074, 77.1026230
4: -30.5670910, 44.1857262, -35.4032402, 52.2151566, -82.7822342, 79.5889511

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9217822, upper bound: 77.9392343
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9217822, upper bound: 77.9392343
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -34.1427422, 40.8300209, -69.0160599, 67.6623001
1: -21.7694626, 26.5227585, -26.2666893, 32.2715721, -54.0410309, 52.7894402
2: -18.9085445, 26.6295929, -22.8611107, 32.1802635, -51.0888062, 49.4907036
3: -25.9847813, 31.7821980, -31.4216194, 38.6731300, -64.6579132, 63.2038193
4: -24.4721470, 35.5742149, -29.6509647, 42.9414139, -67.4135590, 65.2251816

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9306773, upper bound: 77.9478354
time: 1.04 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9503350, upper bound: 77.9486572
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -37.0238686, 43.8689270, -72.0549698, 70.5434418
1: -21.7694626, 26.5227585, -28.3667164, 34.7008591, -56.4703217, 54.8894730
2: -18.9085445, 26.6295929, -24.7050591, 34.5376816, -53.4462242, 51.3346481
3: -25.9847813, 31.7821980, -33.9762878, 41.5369606, -67.5217438, 65.7584839
4: -24.4721470, 35.5742149, -31.9859676, 46.0848083, -70.5569534, 67.5601807

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9507920, upper bound: 77.9335019
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9514980, upper bound: 77.9490651
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -37.0941353, 44.8525009, -73.0385284, 70.6137009
1: -21.7694626, 26.5227585, -28.7098198, 35.5209656, -57.2904282, 55.2325745
2: -18.9085445, 26.6295929, -24.9796867, 35.5278587, -54.4364014, 51.6092796
3: -25.9847813, 31.7821980, -34.3585892, 42.5810547, -68.5658340, 66.1407852
4: -24.4721470, 35.5742149, -32.4492874, 47.4416008, -71.9137497, 68.0234985

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9497616, upper bound: 77.9336547
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9504675, upper bound: 77.9492178
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -39.8363800, 47.8027878, -75.9888306, 73.3559570
1: -21.7694626, 26.5227585, -30.7244854, 37.8837204, -59.6531830, 57.2472382
2: -18.9085445, 26.6295929, -26.7479649, 37.8249779, -56.7335205, 53.3775558
3: -25.9847813, 31.7821980, -36.8228951, 45.3628349, -71.3476181, 68.6050873
4: -24.4721470, 35.5742149, -34.6951561, 50.5066795, -74.9788132, 70.2693710

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9508576, upper bound: 77.9341988
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9515636, upper bound: 77.9497620
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -22.9702396, 26.8798943, -41.1443329, 49.7956924, -72.7659302, 68.0242233
1: -17.5727367, 21.1764984, -31.8737583, 39.4841576, -57.0568924, 53.0502548
2: -15.2716131, 21.2516975, -27.7399025, 39.5278969, -54.7995110, 48.9916000
3: -21.0244789, 25.3388939, -38.2511063, 47.2646751, -68.2891388, 63.5899887
4: -19.7019234, 28.3351135, -36.0435677, 52.7837029, -72.4856262, 64.3786774

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9374717, upper bound: 77.9492546
time: 1.11 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387837, upper bound: 77.9499121
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -41.1443329, 49.7956924, -75.9074097, 71.6041412
1: -19.9750519, 24.0622005, -31.8737583, 39.4841576, -59.4592094, 55.9359589
2: -17.3923473, 24.0895405, -27.7399025, 39.5278969, -56.9202423, 51.8294449
3: -23.9274788, 28.7653179, -38.2511063, 47.2646751, -71.1921463, 67.0164185
4: -22.4313278, 32.1443291, -36.0435677, 52.7837029, -75.2150269, 68.1878967

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9376992, upper bound: 77.9253483
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382799, upper bound: 77.9403890
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -34.8093834, 41.3694382, -38.9889221, 46.9944725, -81.8038559, 80.3583603
1: -26.7821999, 32.6442871, -30.1054592, 37.2317505, -64.0139465, 62.7497482
2: -23.3394756, 32.7738380, -26.2055988, 37.2606354, -60.6000977, 58.9794273
3: -32.3881607, 38.9999428, -36.1685638, 44.5500259, -76.9381790, 75.1684799
4: -30.1711388, 43.8623695, -34.0230713, 49.7402000, -79.9113235, 77.8854370

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9322400, upper bound: 77.9375539
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9325445, upper bound: 77.9379655
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -34.8093834, 41.3694382, -42.3495522, 51.6866341, -86.4960175, 83.7189865
1: -26.7821999, 32.6442871, -32.8885078, 40.9174194, -67.6996078, 65.5327911
2: -23.3394756, 32.7738380, -28.6211510, 41.0825195, -64.4219971, 61.3949699
3: -32.3881607, 38.9999428, -39.6321793, 48.9405556, -81.3287125, 78.6321182
4: -30.1711388, 43.8623695, -37.1985016, 54.9127235, -85.0838623, 81.0608597

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384859, upper bound: 77.9392634
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384722, upper bound: 77.9397099
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -36.1700401, 43.4971924, -35.1277313, 41.9238472, -78.0938797, 78.6249237
1: -27.9390888, 34.4819832, -27.0110054, 33.2081642, -61.1472511, 61.4929886
2: -24.3269978, 34.4814186, -23.5259743, 33.1640472, -57.4910431, 58.0073891
3: -33.5344810, 41.2522278, -32.4333725, 39.7198219, -73.2543030, 73.6856003
4: -31.5612526, 46.0148773, -30.4895687, 44.2436790, -75.8049316, 76.5044403

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9338058, upper bound: 77.9451553
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9338058, upper bound: 77.9451553
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -36.1700401, 43.4971924, -47.1281700, 57.0560150, -93.2260590, 90.6253662
1: -27.9390888, 34.4819832, -36.4152336, 45.1656876, -73.1047668, 70.8972168
2: -24.3269978, 34.4814186, -31.7646389, 45.2699165, -69.5969162, 66.2460556
3: -33.5344810, 41.2522278, -44.1128006, 53.9153290, -87.4498062, 85.3650284
4: -31.5612526, 46.0148773, -41.2324677, 60.5475235, -92.1087799, 87.2473373

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451146
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451553
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -48.3010254, 58.6476707, -35.1277313, 41.9238472, -90.2248688, 93.7754059
1: -37.4080391, 46.4722481, -27.0110054, 33.2081642, -70.6161957, 73.4832535
2: -32.6320572, 46.5963326, -23.5259743, 33.1640472, -65.7960815, 70.1223068
3: -45.2977142, 55.4843941, -32.4333725, 39.7198219, -85.0175323, 87.9177704
4: -42.3785172, 62.3377914, -30.4895687, 44.2436790, -86.6221924, 92.8273621

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9250992, upper bound: 77.9384286
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9334520, upper bound: 77.9384355
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -48.3010254, 58.6476707, -47.1281700, 57.0560150, -105.3570404, 105.7758408
1: -37.4080391, 46.4722481, -36.4152336, 45.1656876, -82.5737152, 82.8874817
2: -32.6320572, 46.5963326, -31.7646389, 45.2699165, -77.9019775, 78.3609695
3: -45.2977142, 55.4843941, -44.1128006, 53.9153290, -99.2130432, 99.5971985
4: -42.3785172, 62.3377914, -41.2324677, 60.5475235, -102.9260406, 103.5702438

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9330271, upper bound: 77.9330271
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9330271, upper bound: 77.9386206
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -38.4355240, 46.6413689, -26.7100945, 31.7491989, -70.1847229, 73.3514481
1: -29.8471088, 36.9797592, -20.5566368, 25.0064430, -54.8535461, 57.5363960
2: -25.9868698, 37.0171318, -17.8831596, 25.1746540, -51.1615219, 54.9002838
3: -35.8958549, 44.2278976, -24.7481346, 29.8912258, -65.7870712, 68.9760208
4: -33.7544556, 49.4227867, -23.0991211, 33.6519928, -67.4064484, 72.5219040

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9463847, upper bound: 77.9228744
time: 1.12 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9463847, upper bound: 77.9255451
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -38.4355240, 46.6413689, -36.9758110, 44.2695045, -82.7050171, 83.6171646
1: -29.8471088, 36.9797592, -28.5059242, 34.8957939, -64.7429047, 65.4856873
2: -25.9868698, 37.0171318, -24.8561287, 35.1255722, -61.1124344, 61.8732605
3: -35.8958549, 44.2278976, -34.6443596, 41.6533089, -77.5491638, 78.8722534
4: -33.7544556, 49.4227867, -32.1453514, 47.0764198, -80.8308716, 81.5681305

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9394825, upper bound: 77.9250954
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9392343, upper bound: 77.9217822
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.1717911, 49.8883514, -26.7100945, 31.7491989, -72.9209900, 76.5984116
1: -31.9071941, 39.5558090, -20.5566368, 25.0064430, -56.9136353, 60.1124458
2: -27.7717896, 39.6101379, -17.8831596, 25.1746540, -52.9464417, 57.4932976
3: -38.3161316, 47.3437538, -24.7481346, 29.8912258, -68.2073441, 72.0918732
4: -36.0879478, 52.8990135, -23.0991211, 33.6519928, -69.7399368, 75.9981003

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9440230, upper bound: 77.9361275
time: 1.17 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459905, upper bound: 77.9381537
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -41.1717911, 49.8883514, -36.9758110, 44.2695045, -85.4412994, 86.8641434
1: -31.9071941, 39.5558090, -28.5059242, 34.8957939, -66.8029861, 68.0617218
2: -27.7717896, 39.6101379, -24.8561287, 35.1255722, -62.8973541, 64.4662628
3: -38.3161316, 47.3437538, -34.6443596, 41.6533089, -79.9694366, 81.9881058
4: -36.0879478, 52.8990135, -32.1453514, 47.0764198, -83.1643600, 85.0443268

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398793, upper bound: 77.9390271
time: 1.06 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9397099, upper bound: 77.9384722
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -36.4147873, 43.7765236, -39.7738800, 48.6438789, -85.0586472, 83.5503998
1: -28.1279392, 34.7059441, -30.9381809, 38.4866905, -66.6146317, 65.6440964
2: -24.4909668, 34.7017326, -26.9366169, 38.6396027, -63.1305695, 61.6383514
3: -33.7604027, 41.5192337, -37.3691788, 45.9937096, -79.7541122, 78.8884125
4: -31.7717438, 46.3106346, -34.9922638, 51.6398849, -83.4116287, 81.3028870

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9252975, upper bound: 77.9376606
time: 1.57 seconds

## Relational analysis of NS_A2_B2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9252975, upper bound: 77.9376606
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -39.8556862, 48.1743584, -42.4002914, 51.8007088, -91.6563873, 90.5746384
1: -30.8398857, 38.1806488, -32.9388084, 41.0035210, -71.8434067, 71.1194611
2: -26.8386936, 38.2200851, -28.6668015, 41.1806908, -68.0193710, 66.8868866
3: -37.0368347, 45.6975822, -39.7164993, 49.0372391, -86.0740662, 85.4140778
4: -34.8641777, 51.0304565, -37.2602005, 55.0491829, -89.9133530, 88.2906570

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387814, upper bound: 77.9453331
time: 1.08 seconds

## Relational analysis of NS_A2_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386641, upper bound: 77.9390293
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -43.0477867, 51.8124046, -42.4002914, 51.8007088, -94.8484879, 94.2126923
1: -33.2721481, 41.0968285, -32.9388084, 41.0035210, -74.2756653, 74.0356369
2: -28.9815655, 41.0912323, -28.6668015, 41.1806908, -70.1622543, 69.7580338
3: -39.9843903, 49.1549530, -39.7164993, 49.0372391, -89.0216293, 88.8714447
4: -37.6193199, 54.8781319, -37.2602005, 55.0491829, -92.6684952, 92.1383362

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382322, upper bound: 77.9355699
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588
time: 1.15 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.90 seconds
NS_A1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9034696, upper bound: 77.9502207
NS_A1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9034696, upper bound: 77.9515701
NS_A1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9062387, upper bound: 77.9499455
NS_A1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9212100, upper bound: 77.9509991
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9067702, upper bound: 77.9365357
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9209037, upper bound: 77.9375893
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9217822, upper bound: 77.9392343
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9217822, upper bound: 77.9392343
NS_A1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9306773, upper bound: 77.9478354
NS_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9503350, upper bound: 77.9486572
NS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9507920, upper bound: 77.9335019
NS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9514980, upper bound: 77.9490651
NS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9497616, upper bound: 77.9336547
NS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9504675, upper bound: 77.9492178
NS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9508576, upper bound: 77.9341988
NS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9515636, upper bound: 77.9497620
NS_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9374717, upper bound: 77.9492546
NS_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9387837, upper bound: 77.9499121
NS_A1_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9376992, upper bound: 77.9253483
NS_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9382799, upper bound: 77.9403890
NS_A1_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9322400, upper bound: 77.9375539
NS_A1_B2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9325445, upper bound: 77.9379655
NS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9384859, upper bound: 77.9392634
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9384722, upper bound: 77.9397099
NS_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9338058, upper bound: 77.9451553
NS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9338058, upper bound: 77.9451553
NS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451146
NS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451553
NS_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9250992, upper bound: 77.9384286
NS_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9334520, upper bound: 77.9384355
NS_A2_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9330271, upper bound: 77.9330271
NS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9330271, upper bound: 77.9386206
NS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9463847, upper bound: 77.9228744
NS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9463847, upper bound: 77.9255451
NS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9394825, upper bound: 77.9250954
NS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9392343, upper bound: 77.9217822
NS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9440230, upper bound: 77.9361275
NS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9459905, upper bound: 77.9381537
NS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9398793, upper bound: 77.9390271
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9397099, upper bound: 77.9384722
NS_A2_B2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9252975, upper bound: 77.9376606
NS_A2_B2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9252975, upper bound: 77.9376606
NS_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9387814, upper bound: 77.9453331
NS_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9386641, upper bound: 77.9390293
NS_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9382322, upper bound: 77.9355699
NS_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588

## BFS NS instance: NS_A1_B1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -21.6248856, 24.6825733, -25.5432625, 30.3792229, -52.0041084, 50.2258377
1: -16.3540230, 19.4223385, -19.7759457, 24.0235004, -40.3775101, 39.1982841
2: -14.2361240, 19.4072628, -17.1907673, 24.1177368, -38.3538589, 36.5980301
3: -19.5663929, 23.2162151, -23.6642838, 28.7449360, -48.3113251, 46.8805008
4: -18.3040810, 25.8403549, -22.2222214, 32.1945114, -50.4985924, 48.0625763

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9034290, upper bound: 77.9501551
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9034073, upper bound: 77.9501468
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -21.6248856, 24.6825733, -38.4355240, 46.6413689, -68.2662506, 63.1180954
1: -16.3540230, 19.4223385, -29.8471088, 36.9797592, -53.3337784, 49.2694473
2: -14.2361240, 19.4072628, -25.9868698, 37.0171318, -51.2532539, 45.3941307
3: -19.5663929, 23.2162151, -35.8958549, 44.2278976, -63.7942886, 59.1120682
4: -18.3040810, 25.8403549, -33.7544556, 49.4227867, -67.7268524, 59.5948067

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9034290, upper bound: 77.9515045
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9034073, upper bound: 77.9514963
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -22.8986530, 26.7912273, -35.9864082, 44.0514946, -66.9501495, 62.7776260
1: -17.5167942, 21.1046333, -28.0938492, 34.9239807, -52.4407730, 49.1984825
2: -15.2222376, 21.1801529, -24.4450645, 35.0142899, -50.2365265, 45.6252174
3: -20.9571171, 25.2526760, -33.7372665, 41.8102303, -62.7673378, 58.9899445
4: -19.6372547, 28.2393932, -31.7980671, 46.7602959, -66.3975525, 60.0374603

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9062387, upper bound: 77.9499455
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9062387, upper bound: 77.9499455
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -22.8986530, 26.7912273, -38.5109100, 46.7036057, -69.6022568, 65.3021393
1: -17.5167942, 21.1046333, -29.9085979, 37.0295601, -54.5463562, 51.0132294
2: -15.2222376, 21.1801529, -26.0360565, 37.0560913, -52.2783279, 47.2162094
3: -20.9571171, 25.2526760, -35.9501534, 44.2971878, -65.2542725, 61.2028275
4: -19.6372547, 28.2393932, -33.8185539, 49.4768410, -69.1140900, 62.0579453

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9203518, upper bound: 77.9508610
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9203097, upper bound: 77.9509542
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.3306732, 41.7020187, -39.0881996, 47.6289024, -82.9595718, 80.7902145
1: -27.1119175, 32.9430580, -30.3426208, 37.6718483, -64.7837677, 63.2856789
2: -23.6637344, 33.0150185, -26.4152794, 37.7954597, -61.4591789, 59.4302940
3: -32.8386917, 39.3118210, -36.6423264, 45.0189972, -77.8576736, 75.9541321
4: -30.5670910, 44.1857262, -34.3026962, 50.5018272, -81.0689163, 78.4884186

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9067566, upper bound: 77.9368652
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9208900, upper bound: 77.9379188
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -35.3306732, 41.7020187, -40.6566315, 49.5017319, -84.8324051, 82.3586502
1: -27.1119175, 32.9430580, -31.5873890, 39.1911392, -66.3030548, 64.5304489
2: -23.6637344, 33.0150185, -27.5211220, 39.2962914, -62.9600143, 60.5361366
3: -32.8386917, 39.3118210, -38.1873322, 46.8023758, -79.6410522, 77.4991379
4: -30.5670910, 44.1857262, -35.7295036, 52.5213623, -83.0884552, 79.9152298

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9067566, upper bound: 77.9368652
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9208900, upper bound: 77.9379188
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -32.3365974, 38.4353638, -66.6213913, 65.8561554
1: -21.7694626, 26.5227585, -24.8443260, 30.3772449, -52.1467056, 51.3670731
2: -18.9085445, 26.6295929, -21.6333523, 30.2357616, -49.1443062, 48.2629471
3: -25.9847813, 31.7821980, -29.7193546, 36.3992424, -62.3840256, 61.5015526
4: -24.4721470, 35.5742149, -28.0216179, 40.3326836, -64.8048172, 63.5958252

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9478354
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9478354
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -28.1860466, 33.5195770, -33.8945045, 40.5032005, -68.6892471, 67.4140778
1: -21.7694626, 26.5227585, -26.0676498, 32.0110092, -53.7804718, 52.5904007
2: -18.9085445, 26.6295929, -22.6887627, 31.9169521, -50.8254929, 49.3183517
3: -25.9847813, 31.7821980, -31.1792412, 38.3591118, -64.3438873, 62.9614410
4: -24.4721470, 35.5742149, -29.4222946, 42.5872154, -67.0593643, 64.9965057

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9501904, upper bound: 77.9486572
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9501904, upper bound: 77.9486572
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -26.2328339, 31.0514660, -37.0238686, 43.8689270, -70.1017609, 68.0753326
1: -20.2322845, 24.5348263, -28.3667164, 34.7008591, -54.9331436, 52.9015427
2: -17.5794239, 24.6057816, -24.7050591, 34.5376816, -52.1171036, 49.3108406
3: -24.1720276, 29.3830109, -33.9762878, 41.5369606, -65.7089844, 63.3592987
4: -22.7215328, 32.8573380, -31.9859676, 46.0848083, -68.8063431, 64.8432922

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9494392, upper bound: 77.9333042
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9494392, upper bound: 77.9332442
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -27.8303680, 33.0590668, -37.0238686, 43.8689270, -71.6992950, 70.0829315
1: -21.4836216, 26.1558895, -28.3667164, 34.7008591, -56.1844749, 54.5226059
2: -18.6609097, 26.2575474, -24.7050591, 34.5376816, -53.1985893, 50.9625931
3: -25.6393242, 31.3390560, -33.9762878, 41.5369606, -67.1762848, 65.3153458
4: -24.1453400, 35.0733643, -31.9859676, 46.0848083, -70.2301483, 67.0593338

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9501905, upper bound: 77.9490651
time: 1.14 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9501905, upper bound: 77.9486571
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -26.2328339, 31.0514660, -37.0941353, 44.8525009, -71.0853271, 68.1455994
1: -20.2322845, 24.5348263, -28.7098198, 35.5209656, -55.7532425, 53.2446442
2: -17.5794239, 24.6057816, -24.9796867, 35.5278587, -53.1072845, 49.5854683
3: -24.1720276, 29.3830109, -34.3585892, 42.5810547, -66.7530823, 63.7416000
4: -22.7215328, 32.8573380, -32.4492874, 47.4416008, -70.1631317, 65.3066177

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9334570
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9336547
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -27.8303680, 33.0590668, -37.0941353, 44.8525009, -72.6828461, 70.1531830
1: -21.4836216, 26.1558895, -28.7098198, 35.5209656, -57.0045815, 54.8657074
2: -18.6609097, 26.2575474, -24.9796867, 35.5278587, -54.1887665, 51.2372208
3: -25.6393242, 31.3390560, -34.3585892, 42.5810547, -68.2203827, 65.6976471
4: -24.1453400, 35.0733643, -32.4492874, 47.4416008, -71.5869446, 67.5226517

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9492178
time: 0.99 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9492178
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -26.2328339, 31.0514660, -39.8363800, 47.8027878, -74.0356216, 70.8878479
1: -20.2322845, 24.5348263, -30.7244854, 37.8837204, -58.1160049, 55.2593117
2: -17.5794239, 24.6057816, -26.7479649, 37.8249779, -55.4044037, 51.3537445
3: -24.1720276, 29.3830109, -36.8228951, 45.3628349, -69.5348663, 66.2058945
4: -22.7215328, 32.8573380, -34.6951561, 50.5066795, -73.2282028, 67.5524750

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9486145, upper bound: 77.9333066
time: 0.82 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9340011
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9340224
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -27.8303680, 33.0590668, -39.8363800, 47.8027878, -75.6331558, 72.8954391
1: -21.4836216, 26.1558895, -30.7244854, 37.8837204, -59.3673401, 56.8803749
2: -18.6609097, 26.2575474, -26.7479649, 37.8249779, -56.4858818, 53.0055122
3: -25.6393242, 31.3390560, -36.8228951, 45.3628349, -71.0021591, 68.1619492
4: -24.1453400, 35.0733643, -34.6951561, 50.5066795, -74.6520081, 69.7685165

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493071, upper bound: 77.9488156
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9497620
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9492178
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -21.6248856, 24.6825733, -41.1443329, 49.7956924, -71.4205780, 65.8269043
1: -16.3540230, 19.4223385, -31.8737583, 39.4841576, -55.8381767, 51.2960968
2: -14.2361240, 19.4072628, -27.7399025, 39.5278969, -53.7640228, 47.1471634
3: -19.5663929, 23.2162151, -38.2511063, 47.2646751, -66.8310699, 61.4673233
4: -18.3040810, 25.8403549, -36.0435677, 52.7837029, -71.0877838, 61.8839226

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9374717, upper bound: 77.9492187
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9374717, upper bound: 77.9492546
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -22.8986530, 26.7912273, -41.1443329, 49.7956924, -72.6943436, 67.9355469
1: -17.5167942, 21.1046333, -31.8737583, 39.4841576, -57.0009537, 52.9783936
2: -15.2222376, 21.1801529, -27.7399025, 39.5278969, -54.7501335, 48.9200554
3: -20.9571171, 25.2526760, -38.2511063, 47.2646751, -68.2217636, 63.5037766
4: -19.6372547, 28.2393932, -36.0435677, 52.7837029, -72.4209595, 64.2829590

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_A1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378479, upper bound: 77.9333939
time: 1.10 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385935, upper bound: 77.9496000
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -25.7951889, 30.0408707, -41.1443329, 49.7956924, -75.5908813, 71.1852036
1: -19.7205257, 23.7264099, -31.8737583, 39.4841576, -59.2046814, 55.6001663
2: -17.1720085, 23.7482529, -27.7399025, 39.5278969, -56.6999054, 51.4881554
3: -23.6165218, 28.3605232, -38.2511063, 47.2646751, -70.8811874, 66.6116257
4: -22.1386547, 31.6833401, -36.0435677, 52.7837029, -74.9223557, 67.7269058

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9323595, upper bound: 77.9385224
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9371148, upper bound: 77.9391054
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -33.4673347, 39.6504440, -42.3495522, 51.6866341, -85.1539688, 82.0000000
1: -25.7021122, 31.2593193, -32.8885078, 40.9174194, -66.6195221, 64.1478271
2: -22.3965511, 31.3766556, -28.6211510, 41.0825195, -63.4790688, 59.9977951
3: -31.1130371, 37.3390770, -39.6321793, 48.9405556, -80.0535889, 76.9712524
4: -28.9364624, 41.9842148, -37.1985016, 54.9127235, -83.8491821, 79.1827011

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9326145, upper bound: 77.9374823
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9373208, upper bound: 77.9380649
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -35.3306732, 41.7020187, -42.3495522, 51.6866341, -87.0173035, 84.0515671
1: -27.1119175, 32.9430580, -32.8885078, 40.9174194, -68.0293350, 65.8315659
2: -23.6637344, 33.0150185, -28.6211510, 41.0825195, -64.7462387, 61.6361656
3: -32.8386917, 39.3118210, -39.6321793, 48.9405556, -81.7792511, 78.9440002
4: -30.5670910, 44.1857262, -37.1985016, 54.9127235, -85.4798126, 81.3842239

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9326008, upper bound: 77.9378117
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9373071, upper bound: 77.9383944
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -34.4260216, 41.1059990, -35.1277313, 41.9238472, -76.3498688, 76.2337265
1: -26.4669743, 32.5521507, -27.0110054, 33.2081642, -59.6751366, 59.5631561
2: -23.0558872, 32.5177460, -23.5259743, 33.1640472, -56.2199326, 56.0437202
3: -31.7907181, 38.9283676, -32.4333725, 39.7198219, -71.5105438, 71.3617401
4: -29.8766003, 43.3741875, -30.4895687, 44.2436790, -74.1202774, 73.8637543

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9523003, upper bound: 77.9458435
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9513972, upper bound: 77.9459499
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -38.1909790, 46.3809471, -35.1277313, 41.9238472, -80.1148224, 81.5086823
1: -29.5813637, 36.6850815, -27.0110054, 33.2081642, -62.7895279, 63.6960869
2: -25.7531719, 36.8137054, -23.5259743, 33.1640472, -58.9172211, 60.3396797
3: -35.6783218, 43.8473434, -32.4333725, 39.7198219, -75.3981476, 76.2807159
4: -33.4244843, 49.1868057, -30.4895687, 44.2436790, -77.6681671, 79.6763611

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9374670, upper bound: 77.9453711
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9520893, upper bound: 77.9457626
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -36.1700401, 43.4971924, -44.2251472, 53.5711441, -89.7411652, 87.7223282
1: -27.9390888, 34.4819832, -34.1928864, 42.3872795, -70.3263702, 68.6748428
2: -24.3269978, 34.4814186, -29.8444653, 42.4670563, -66.7940369, 64.3258820
3: -33.5344810, 41.2522278, -41.5141678, 50.5573082, -84.0917892, 82.7663956
4: -31.5612526, 46.0148773, -38.7179718, 56.7911415, -88.3523941, 84.7328491

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451146
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9225479, upper bound: 77.9274418
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -36.1700401, 43.4971924, -46.8785782, 56.7771034, -92.9471359, 90.3757706
1: -27.9390888, 34.4819832, -36.2231636, 44.9410744, -72.8801498, 70.7051315
2: -24.3269978, 34.4814186, -31.5978546, 45.0498505, -69.3768387, 66.0792694
3: -33.5344810, 41.2522278, -43.8842010, 53.6473083, -87.1817932, 85.1364288
4: -31.5612526, 46.0148773, -41.0190010, 60.2521362, -91.8133850, 87.0338745

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451554
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9225479, upper bound: 77.9451146
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -48.3010254, 58.6476707, -33.1754951, 39.4201012, -87.7211304, 91.8231659
1: -37.4080391, 46.4722481, -25.4743347, 31.2093258, -68.6173630, 71.9465790
2: -32.6320572, 46.5963326, -22.1972332, 31.1237240, -63.7557716, 68.7935638
3: -45.2977142, 55.4843941, -30.6023026, 37.3121796, -82.6098862, 86.0866928
4: -42.3785172, 62.3377914, -28.7347183, 41.5055771, -83.8840866, 91.0725021

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9364487, upper bound: 77.9273204
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9317892, upper bound: 77.9369131
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9367076, upper bound: 77.9377398
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -48.3010254, 58.6476707, -34.8070755, 41.4969826, -89.7980042, 93.4547119
1: -37.4080391, 46.4722481, -26.7536716, 32.8676605, -70.2756882, 73.2259216
2: -32.6320572, 46.5963326, -23.3026772, 32.8174057, -65.4494553, 69.8989944
3: -45.2977142, 55.4843941, -32.1163902, 39.3115959, -84.6093140, 87.6007843
4: -42.3785172, 62.3377914, -30.1930466, 43.7772369, -86.1557312, 92.5308304

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9500168, upper bound: 77.9377344
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9512639, upper bound: 77.9381314
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -49.5578003, 60.5888596, -47.1281700, 57.0560150, -106.6138153, 107.7170258
1: -38.4581070, 47.9387932, -36.4152336, 45.1656876, -83.6237946, 84.3540268
2: -33.5468063, 48.1905899, -31.7646389, 45.2699165, -78.8167267, 79.9552307
3: -46.7278366, 57.2010765, -44.1128006, 53.9153290, -100.6431427, 101.3138733
4: -43.5837936, 64.5246735, -41.2324677, 60.5475235, -104.1313171, 105.7571411

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9244637, upper bound: 77.9384286
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9328165, upper bound: 77.9384355
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -36.4300194, 43.9885750, -26.7100945, 31.7491989, -68.1792145, 70.6986542
1: -28.1788616, 34.8463860, -20.5566368, 25.0064430, -53.1853027, 55.4030228
2: -24.5429592, 34.8592529, -17.8831596, 25.1746540, -49.7176132, 52.7424126
3: -33.9304657, 41.6542892, -24.7481346, 29.8912258, -63.8216934, 66.4024200
4: -31.8499851, 46.5238953, -23.0991211, 33.6519928, -65.5019760, 69.6230164

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9442654, upper bound: 77.9133959
time: 1.06 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9453183, upper bound: 77.9220175
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -39.7738800, 48.6438789, -26.7100945, 31.7491989, -71.5230789, 75.3539581
1: -30.9381809, 38.4866905, -20.5566368, 25.0064430, -55.9446220, 59.0433273
2: -26.9366169, 38.6396027, -17.8831596, 25.1746540, -52.1112671, 56.5227585
3: -37.3691788, 45.9937096, -24.7481346, 29.8912258, -67.2603989, 70.7418289
4: -34.9922638, 51.6398849, -23.0991211, 33.6519928, -68.6442566, 74.7390060

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9442654, upper bound: 77.9133959
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9453183, upper bound: 77.9249051
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -36.9522400, 44.7341309, -36.9758110, 44.2695045, -81.2217331, 81.7099228
1: -28.6438560, 35.4471130, -28.5059242, 34.8957939, -63.5396500, 63.9530296
2: -24.9379539, 35.4695740, -24.8561287, 35.1255722, -60.0635262, 60.3256989
3: -34.4548569, 42.3916130, -34.6443596, 41.6533089, -76.1081696, 77.0359650
4: -32.3821983, 47.3406982, -32.1453514, 47.0764198, -79.4586029, 79.4860535

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9370192, upper bound: 77.9130979
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381670, upper bound: 77.9244448
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.0499535, 48.2958298, -36.9758110, 44.2695045, -84.3194580, 85.2716370
1: -31.0301151, 38.3110580, -28.5059242, 34.8957939, -65.9259033, 66.8169861
2: -27.0399628, 38.2955475, -24.8561287, 35.1255722, -62.1655350, 63.1516762
3: -37.3484612, 45.7857323, -34.6443596, 41.6533089, -79.0017624, 80.4300842
4: -35.0852585, 51.1316071, -32.1453514, 47.0764198, -82.1616821, 83.2769165

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387878, upper bound: 77.9217822
time: 0.90 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387878, upper bound: 77.9217822
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -38.1943321, 46.6883850, -26.7100945, 31.7491989, -69.9435272, 73.3984680
1: -29.7238026, 36.9951706, -20.5566368, 25.0064430, -54.7302399, 57.5518074
2: -25.8526402, 37.1122932, -17.8831596, 25.1746540, -51.0272942, 54.9954529
3: -35.6520920, 44.3149033, -24.7481346, 29.8912258, -65.5433121, 69.0630341
4: -33.6420479, 49.5697975, -23.0991211, 33.6519928, -67.2940369, 72.6689148

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9449473, upper bound: 77.9324605
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9389110, upper bound: 77.9358890
time: 0.98 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387299, upper bound: 77.9326011
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -40.7746887, 49.4042358, -26.7100945, 31.7491989, -72.5238876, 76.1143188
1: -31.6008244, 39.1633835, -20.5566368, 25.0064430, -56.6072617, 59.7200203
2: -27.5028191, 39.2167053, -17.8831596, 25.1746540, -52.6774750, 57.0998611
3: -37.9414215, 46.8796577, -24.7481346, 29.8912258, -67.8326492, 71.6277924
4: -35.7357635, 52.3725014, -23.0991211, 33.6519928, -69.3877487, 75.4716187

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9453589, upper bound: 77.9327650
time: 1.10 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9453589, upper bound: 77.9381537
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -39.5844460, 47.8753853, -36.9758110, 44.2695045, -83.8539200, 84.8511734
1: -30.6317654, 37.9366074, -28.5059242, 34.8957939, -65.5275574, 66.4425354
2: -26.6574287, 37.9808197, -24.8561287, 35.1255722, -61.7829971, 62.8369484
3: -36.7881088, 45.4067116, -34.6443596, 41.6533089, -78.4414215, 80.0510712
4: -34.6318779, 50.7094955, -32.1453514, 47.0764198, -81.7082901, 82.8548279

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9379928, upper bound: 77.9358887
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385638, upper bound: 77.9379149
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.7816925, 51.5114365, -36.9758110, 44.2695045, -87.0511856, 88.4872437
1: -33.0620728, 40.8505859, -28.5059242, 34.8957939, -67.9578705, 69.3565063
2: -28.7987938, 40.8499374, -24.8561287, 35.1255722, -63.9243622, 65.7060699
3: -39.7337646, 48.8617172, -34.6443596, 41.6533089, -81.3870544, 83.5060730
4: -37.3851776, 54.5544510, -32.1453514, 47.0764198, -84.4615936, 86.6997833

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9368652, upper bound: 77.9326008
time: 1.08 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383944, upper bound: 77.9373071
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -35.1231308, 42.0763168, -42.4002914, 51.8007088, -86.9238205, 84.4765930
1: -27.0699730, 33.3366203, -32.9388084, 41.0035210, -68.0734940, 66.2754288
2: -23.5692272, 33.3143387, -28.6668015, 41.1806908, -64.7499161, 61.9811401
3: -32.4885712, 39.8816452, -39.7164993, 49.0372391, -81.5258026, 79.5981445
4: -30.5635567, 44.4425621, -37.2602005, 55.0491829, -85.6127319, 81.7027588

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9387871
time: 1.10 seconds

## Relational analysis of NS_A2_B2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9390293
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -47.1078300, 57.0899124, -42.4002914, 51.8007088, -98.9085312, 99.4901962
1: -36.4343338, 45.2124557, -32.9388084, 41.0035210, -77.4378510, 78.1512604
2: -31.7808056, 45.3215141, -28.6668015, 41.1806908, -72.9614944, 73.9883118
3: -44.1263847, 53.9756012, -39.7164993, 49.0372391, -93.1635971, 93.6920853
4: -41.2637596, 60.6202011, -37.2602005, 55.0491829, -96.3129196, 97.8804016

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9387871
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9390293
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -38.3238754, 45.7305298, -42.4002914, 51.8007088, -90.1245880, 88.1308136
1: -29.5014935, 36.2605743, -32.9388084, 41.0035210, -70.5050125, 69.1993866
2: -25.7097168, 36.1988602, -28.6668015, 41.1806908, -66.8904037, 64.8656540
3: -35.4347000, 43.3503380, -39.7164993, 49.0372391, -84.4719315, 83.0668335
4: -33.3164864, 48.3035774, -37.2602005, 55.0491829, -88.3656693, 85.5637741

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9353433, upper bound: 77.9353434
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9353433, upper bound: 77.9355699
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -42.4002914, 51.8007088, -100.7230911, 101.5065765
1: -37.8203354, 46.8542061, -32.9388084, 41.0035210, -78.8238525, 79.7930145
2: -33.0129776, 46.9222374, -28.6668015, 41.1806908, -74.1936646, 75.5890350
3: -45.8136063, 55.9145012, -39.7164993, 49.0372391, -94.8508453, 95.6309967
4: -42.8412704, 62.7641792, -37.2602005, 55.0491829, -97.8904266, 100.0243835

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588
time: 0.97 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588
time: 1.23 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.02 seconds
NS_A1_B1_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9034290, upper bound: 77.9501551
NS_A1_B1_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9034073, upper bound: 77.9501468
NS_A1_B1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9034290, upper bound: 77.9515045
NS_A1_B1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9034073, upper bound: 77.9514963
NS_A1_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9062387, upper bound: 77.9499455
NS_A1_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9062387, upper bound: 77.9499455
NS_A1_B1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9203518, upper bound: 77.9508610
NS_A1_B1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9203097, upper bound: 77.9509542
NS_A1_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9067566, upper bound: 77.9368652
NS_A1_B1_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9208900, upper bound: 77.9379188
NS_A1_B1_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9067566, upper bound: 77.9368652
NS_A1_B1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9208900, upper bound: 77.9379188
NS_A1_B2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9478354
NS_A1_B2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9478354
NS_A1_B2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9501904, upper bound: 77.9486572
NS_A1_B2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9501904, upper bound: 77.9486572
NS_A1_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9494392, upper bound: 77.9333042
NS_A1_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9494392, upper bound: 77.9332442
NS_A1_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9501905, upper bound: 77.9490651
NS_A1_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9501905, upper bound: 77.9486571
NS_A1_B2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9334570
NS_A1_B2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9336547
NS_A1_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9492178
NS_A1_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9492178
NS_A1_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9340011
NS_A1_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9495717, upper bound: 77.9340224
NS_A1_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9497620
NS_A1_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9503230, upper bound: 77.9492178
NS_A1_B2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9374717, upper bound: 77.9492187
NS_A1_B2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9374717, upper bound: 77.9492546
NS_A1_B2_B2_A1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9378479, upper bound: 77.9333939
NS_A1_B2_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9385935, upper bound: 77.9496000
NS_A1_B2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9323595, upper bound: 77.9385224
NS_A1_B2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9371148, upper bound: 77.9391054
NS_A1_B2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9326145, upper bound: 77.9374823
NS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9373208, upper bound: 77.9380649
NS_A1_B2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9326008, upper bound: 77.9378117
NS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9373071, upper bound: 77.9383944
NS_A2_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9523003, upper bound: 77.9458435
NS_A2_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9513972, upper bound: 77.9459499
NS_A2_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9374670, upper bound: 77.9453711
NS_A2_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9520893, upper bound: 77.9457626
NS_A2_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451146
NS_A2_B1_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9225479, upper bound: 77.9274418
NS_A2_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9230078, upper bound: 77.9451554
NS_A2_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9225479, upper bound: 77.9451146
NS_A2_B1_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9317892, upper bound: 77.9369131
NS_A2_B1_A2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9367076, upper bound: 77.9377398
NS_A2_B1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9500168, upper bound: 77.9377344
NS_A2_B1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9512639, upper bound: 77.9381314
NS_A2_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9244637, upper bound: 77.9384286
NS_A2_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9328165, upper bound: 77.9384355
NS_A2_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9442654, upper bound: 77.9133959
NS_A2_B2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9453183, upper bound: 77.9220175
NS_A2_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9442654, upper bound: 77.9133959
NS_A2_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9453183, upper bound: 77.9249051
NS_A2_B2_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9370192, upper bound: 77.9130979
NS_A2_B2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9381670, upper bound: 77.9244448
NS_A2_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9387878, upper bound: 77.9217822
NS_A2_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9387878, upper bound: 77.9217822
NS_A2_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9389110, upper bound: 77.9358890
NS_A2_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9387299, upper bound: 77.9326011
NS_A2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9453589, upper bound: 77.9327650
NS_A2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9453589, upper bound: 77.9381537
NS_A2_B2_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9379928, upper bound: 77.9358887
NS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9385638, upper bound: 77.9379149
NS_A2_B2_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9368652, upper bound: 77.9326008
NS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9383944, upper bound: 77.9373071
NS_A2_B2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9387871
NS_A2_B2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9390293
NS_A2_B2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9387871
NS_A2_B2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9357752, upper bound: 77.9390293
NS_A2_B2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9353433, upper bound: 77.9353434
NS_A2_B2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9353433, upper bound: 77.9355699
NS_A2_B2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588
NS_A2_B2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.02
Output dim: 3, lower bound: -77.9384946, upper bound: 77.9384588

## BFS NS instance: NS_A1_B1_A1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -19.0223427, 21.0425797, -25.5432625, 30.3792229, -49.4015579, 46.5858421
1: -14.1951771, 16.5293064, -19.7759457, 24.0235004, -38.2186661, 36.3052521
2: -12.3786316, 16.4371872, -17.1907673, 24.1177368, -36.4963684, 33.6279526
3: -16.9926472, 19.7202187, -23.6642838, 28.7449360, -45.7375755, 43.3845024
4: -15.8382807, 21.8436413, -22.2222214, 32.1945114, -48.0327911, 44.0658646

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8935223, upper bound: 77.9487169
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8936537, upper bound: 77.9493300
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -21.5895576, 24.6392212, -25.5432625, 30.3792229, -51.9687805, 50.1824837
1: -16.3267136, 19.3868980, -19.7759457, 24.0235004, -40.3502007, 39.1628418
2: -14.2115555, 19.3722286, -17.1907673, 24.1177368, -38.3292923, 36.5629959
3: -19.5339737, 23.1740227, -23.6642838, 28.7449360, -48.2789001, 46.8383064
4: -18.2724152, 25.7940540, -22.2222214, 32.1945114, -50.4669266, 48.0162735

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8935232, upper bound: 77.9486928
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8936319, upper bound: 77.9493148
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -19.0223427, 21.0425797, -38.4355240, 46.6413689, -65.6637039, 59.4781036
1: -14.1951771, 16.5293064, -29.8471088, 36.9797592, -51.1749344, 46.3764076
2: -12.3786316, 16.4371872, -25.9868698, 37.0171318, -49.3957596, 42.4240570
3: -16.9926472, 19.7202187, -35.8958549, 44.2278976, -61.2205429, 55.6160736
4: -15.8382807, 21.8436413, -33.7544556, 49.4227867, -65.2610626, 55.5980988

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8935223, upper bound: 77.9502775
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9203315, upper bound: 77.9508907
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -21.5895576, 24.6392212, -38.4355240, 46.6413689, -68.2309265, 63.0747414
1: -16.3267136, 19.3868980, -29.8471088, 36.9797592, -53.3064690, 49.2340012
2: -14.2115555, 19.3722286, -25.9868698, 37.0171318, -51.2286797, 45.3591003
3: -19.5339737, 23.1740227, -35.8958549, 44.2278976, -63.7618599, 59.0698776
4: -18.2724152, 25.7940540, -33.7544556, 49.4227867, -67.6951904, 59.5485077

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9202010, upper bound: 77.9502534
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_A1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9202010, upper bound: 77.9508755
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -22.8986530, 26.7912273, -34.6719971, 42.3210831, -65.2197342, 61.4632263
1: -17.5167942, 21.1046333, -27.0123634, 33.5315552, -51.0483475, 48.1169930
2: -15.2222376, 21.1801529, -23.5037670, 33.6024742, -48.8247108, 44.6839218
3: -20.9571171, 25.2526760, -32.4474258, 40.1424599, -61.0995789, 57.7001038
4: -19.6372547, 28.2393932, -30.5649395, 44.8625908, -64.4998398, 58.8043327

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9061155, upper bound: 77.9493104
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9069286, upper bound: 77.9499455
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -22.8986530, 26.7912273, -37.6383781, 45.7751617, -68.6738129, 64.4295883
1: -17.5167942, 21.1046333, -29.3122444, 36.3085556, -53.8253479, 50.4168739
2: -15.2222376, 21.1801529, -25.5279121, 36.3475189, -51.5697556, 46.7080650
3: -20.9571171, 25.2526760, -35.2353554, 43.4323082, -64.3894196, 60.4880295
4: -19.6372547, 28.2393932, -33.1700668, 48.5421753, -68.1794205, 61.4094582

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9069286, upper bound: 77.9493104
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9061155, upper bound: 77.9499455
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19.9392662, 22.6418819, -38.5109100, 46.7036057, -66.6428680, 61.1527901
1: -15.0526571, 17.7766361, -29.9085979, 37.0295601, -52.0822144, 47.6852341
2: -13.0902519, 17.7617302, -26.0360565, 37.0560913, -50.1463394, 43.7977867
3: -18.0218525, 21.2459145, -35.9501534, 44.2971878, -62.3190346, 57.1960640
4: -16.8131828, 23.6301174, -33.8185539, 49.4768410, -66.2900162, 57.4486694

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8936740, upper bound: 77.9493003
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.8936740, upper bound: 77.9508610
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -22.8519516, 26.7354374, -38.5109100, 46.7036057, -69.5555573, 65.2463455
1: -17.4800701, 21.0594463, -29.9085979, 37.0295601, -54.5096283, 50.9680443
2: -15.1898499, 21.1357193, -26.0360565, 37.0560913, -52.2459335, 47.1717682
3: -20.9132805, 25.1986847, -35.9501534, 44.2971878, -65.2104416, 61.1488380
4: -19.5953865, 28.1802998, -33.8185539, 49.4768410, -69.0722275, 61.9988480

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9069286, upper bound: 77.9503383
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_A1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9069286, upper bound: 77.9500142
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -25.7905121, 30.8886795, -32.3365974, 38.4353638, -64.2258606, 63.2252617
1: -20.0136223, 24.4318047, -24.8443260, 30.3772449, -50.3908653, 49.2761269
2: -17.3697777, 24.5828743, -21.6333523, 30.2357616, -47.6055374, 46.2162209
3: -23.8240852, 29.3207245, -29.7193546, 36.3992424, -60.2233276, 59.0400620
4: -22.4987125, 32.8319511, -28.0216179, 40.3326836, -62.8313980, 60.8535690

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9320745
time: 1.12 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9478354
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -27.8517685, 33.1503563, -32.3365974, 38.4353638, -66.2871170, 65.4869461
1: -21.5249481, 26.2254677, -24.8443260, 30.3772449, -51.9021912, 51.0697899
2: -18.6928768, 26.3362236, -21.6333523, 30.2357616, -48.9286385, 47.9695740
3: -25.6920853, 31.4304199, -29.7193546, 36.3992424, -62.0913277, 61.1497612
4: -24.1958656, 35.1850052, -28.0216179, 40.3326836, -64.5285339, 63.2066231

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9322721
time: 1.03 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9478354
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -25.7905121, 30.8886795, -33.8945045, 40.5032005, -66.2937164, 64.7831802
1: -20.0136223, 24.4318047, -26.0676498, 32.0110092, -52.0246315, 50.4994507
2: -17.3697777, 24.5828743, -22.6887627, 31.9169521, -49.2867279, 47.2716293
3: -23.8240852, 29.3207245, -31.1792412, 38.3591118, -62.1831970, 60.4999619
4: -22.4987125, 32.8319511, -29.4222946, 42.5872154, -65.0859299, 62.2542458

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9328963
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9320745
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -27.8517685, 33.1503563, -33.8945045, 40.5032005, -68.3549652, 67.0448608
1: -21.5249481, 26.2254677, -26.0676498, 32.0110092, -53.5359573, 52.2931137
2: -18.6928768, 26.3362236, -22.6887627, 31.9169521, -50.6098289, 49.0249786
3: -25.6920853, 31.4304199, -31.1792412, 38.3591118, -64.0511932, 62.6096611
4: -24.1958656, 35.1850052, -29.4222946, 42.5872154, -66.7830811, 64.6072998

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9330940
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9486540
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -23.8790150, 28.4682655, -37.0238686, 43.8689270, -67.7479401, 65.4921265
1: -18.4997654, 22.4907513, -28.3667164, 34.7008591, -53.2006226, 50.8574677
2: -16.0649014, 22.5969181, -24.7050591, 34.5376816, -50.6025810, 47.3019753
3: -22.0480022, 26.9633331, -33.9762878, 41.5369606, -63.5849609, 60.9396172
4: -20.7754269, 30.1706944, -31.9859676, 46.0848083, -66.8602371, 62.1566620

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9327997
time: 1.26 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9327997
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -25.9036980, 30.6847572, -37.0238686, 43.8689270, -69.7726212, 67.7086258
1: -19.9909458, 24.2390404, -28.3667164, 34.7008591, -54.6917953, 52.6057587
2: -17.3666553, 24.3128147, -24.7050591, 34.5376816, -51.9043312, 49.0178680
3: -23.8826122, 29.0331535, -33.9762878, 41.5369606, -65.4195633, 63.0094414
4: -22.4487991, 32.4686737, -31.9859676, 46.0848083, -68.5336075, 64.4546432

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9324378
time: 0.96 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9297815, upper bound: 77.9332442
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -25.4711037, 30.4690208, -37.0238686, 43.8689270, -69.3400269, 67.4928818
1: -19.7565899, 24.0980587, -28.3667164, 34.7008591, -54.4574509, 52.4647751
2: -17.1469364, 24.2440948, -24.7050591, 34.5376816, -51.6846161, 48.9491501
3: -23.5127525, 28.9179382, -33.9762878, 41.5369606, -65.0497131, 62.8942261
4: -22.2039375, 32.3764572, -31.9859676, 46.0848083, -68.2887421, 64.3624191

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9485605
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9490424
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -27.4940605, 32.6877251, -37.0238686, 43.8689270, -71.3629913, 69.7115936
1: -21.2376080, 25.8567467, -28.3667164, 34.7008591, -55.9384613, 54.2234650
2: -18.4439354, 25.9623070, -24.7050591, 34.5376816, -52.9816170, 50.6673584
3: -25.3447857, 30.9850636, -33.9762878, 41.5369606, -66.8817444, 64.9613495
4: -23.8673878, 34.6815987, -31.9859676, 46.0848083, -69.9521866, 66.6675644

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9478353
time: 0.89 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305328, upper bound: 77.9486540
time: 0.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.10 + 416.90 = 421.00 seconds
