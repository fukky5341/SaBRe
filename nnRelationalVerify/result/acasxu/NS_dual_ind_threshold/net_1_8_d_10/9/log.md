## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 17.859030126


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.4451218, 9.4399090, -11.4451218, 9.4399090, -20.8850307, 20.8850288)
1: (-45.0262299, 36.2671356, -45.0262299, 36.2671356, -81.2933655, 81.2933655)
2: (-21.7135849, 33.5321465, -21.7135849, 33.5321465, -55.2457237, 55.2457237)
3: (-39.4077721, 32.7506485, -39.4077721, 32.7506485, -72.1584091, 72.1584091)
4: (-28.8892536, 34.1174583, -28.8892536, 34.1174583, -63.0067139, 63.0067139)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.08 + 1.86 = 2.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -17.9037896, upper bound: 17.9037896

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8396848, upper bound: 17.8873289
time: 0.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9029812, upper bound: 17.9029812
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.36 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 0, lower bound: -17.8396848, upper bound: 17.8873289
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 0, lower bound: -17.9029812, upper bound: 17.9029812

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.3769226, 7.8706694, -10.6588011, 8.8387661, -18.2156887, 18.5294685
1: -36.8626022, 30.6436653, -41.8999557, 34.0679970, -70.9305954, 72.5436172
2: -17.8360786, 27.5716228, -20.2405167, 31.2743034, -49.1103821, 47.8121414
3: -32.1838112, 27.6515179, -36.6501999, 30.7681427, -62.9519463, 64.3017197
4: -23.6509323, 28.3754482, -26.8951664, 31.9089184, -55.5598488, 55.2706108

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8382948, upper bound: 17.8382948
time: 0.57 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8382948, upper bound: 17.8873289
time: 0.57 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.5922689, 8.7929964, -11.3493977, 9.3657455, -19.9580116, 20.1423874
1: -41.6339569, 33.9156227, -44.6466560, 35.9975891, -77.6315460, 78.5622787
2: -20.1479568, 31.0970783, -21.5354939, 33.2499237, -53.3978729, 52.6325722
3: -36.4060173, 30.6280861, -39.0733376, 32.5060501, -68.9120636, 69.7014160
4: -26.7219658, 31.7532234, -28.6471825, 33.8342438, -60.5562096, 60.4004021

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8873289, upper bound: 17.8396848
time: 0.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8873289, upper bound: 17.9029812
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.38 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.38
Output dim: 0, lower bound: -17.8382948, upper bound: 17.8382948
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -17.8382948, upper bound: 17.8873289
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -17.8873289, upper bound: 17.8396848
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -17.8873289, upper bound: 17.9029812

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.3769226, 7.8706694, -10.5910549, 8.7920513, -18.1689739, 18.4617233
1: -36.8626022, 30.6436653, -41.6289139, 33.9120827, -70.7746887, 72.2725677
2: -17.8360786, 27.5716228, -20.1456642, 31.0936203, -48.9296951, 47.7172852
3: -32.1838112, 27.6515179, -36.4016762, 30.6249504, -62.8087502, 64.0531921
4: -23.6509323, 28.3754482, -26.7188892, 31.7499275, -55.4008522, 55.0943375

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8068841, upper bound: 17.8692083
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8068841, upper bound: 17.8873289
time: 0.75 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.5922689, 8.7929964, -9.3769226, 7.8706694, -18.4629364, 18.1699181
1: -41.6339569, 33.9156227, -36.8626022, 30.6436653, -72.2776184, 70.7782211
2: -20.1479568, 31.0970783, -17.8360786, 27.5716228, -47.7195663, 48.9331551
3: -36.4060173, 30.6280861, -32.1838112, 27.6515179, -64.0575333, 62.8118935
4: -26.7219658, 31.7532234, -23.6509323, 28.3754482, -55.0974121, 55.4041519

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8731876, upper bound: 17.8336619
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8873289, upper bound: 17.8395697
time: 0.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.5922689, 8.7929964, -10.5922689, 8.7929964, -19.3852596, 19.3852596
1: -41.6339569, 33.9156227, -41.6339569, 33.9156227, -75.5495758, 75.5495758
2: -20.1479568, 31.0970783, -20.1479568, 31.0970783, -51.2450218, 51.2450218
3: -36.4060173, 30.6280861, -36.4060173, 30.6280861, -67.0341034, 67.0341034
4: -26.7219658, 31.7532234, -26.7219658, 31.7532234, -58.4751892, 58.4751892

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8731877, upper bound: 17.9027093
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8873289, upper bound: 17.9027121
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.85 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8068841, upper bound: 17.8692083
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8068841, upper bound: 17.8873289
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8731876, upper bound: 17.8336619
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8873289, upper bound: 17.8395697
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8731877, upper bound: 17.9027093
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -17.8873289, upper bound: 17.9027121

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.2072668, 6.9692569, -10.3218880, 8.5799026, -16.7871685, 17.2911434
1: -32.2548981, 27.3039436, -40.5616798, 33.1231194, -65.3780136, 67.8656158
2: -15.6097517, 24.1825752, -19.6370163, 30.3212185, -45.9309578, 43.8195915
3: -28.1271381, 24.6478672, -35.4595146, 29.9217396, -58.0488777, 60.1073837
4: -20.6859665, 25.1434441, -26.0334129, 30.9995728, -51.6855354, 51.1768532

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8062908, upper bound: 17.8692083
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8062908, upper bound: 17.8692083
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.7493315, 7.3718662, -10.0871792, 8.3898401, -17.1391697, 17.4590454
1: -34.3156853, 28.8483829, -39.6141396, 32.3487854, -66.6644745, 68.4625168
2: -16.8087482, 25.7330456, -19.1885662, 29.6788445, -46.4875832, 44.9216118
3: -29.8720760, 26.0954170, -34.6299553, 29.2403488, -59.1124268, 60.7253723
4: -22.0263462, 26.7866917, -25.4263000, 30.3402138, -52.3665619, 52.2129898

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8145100, upper bound: 17.8767008
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8130413, upper bound: 17.8239444
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.5109835, 7.9441323, -9.0814953, 7.6406231, -17.1516037, 17.0256271
1: -37.3659744, 30.7502995, -35.6932373, 29.7947903, -67.1607590, 66.4435349
2: -18.1058884, 27.9925270, -17.2675304, 26.7085991, -44.8144875, 45.2600555
3: -32.6438370, 27.8008327, -31.1554871, 26.8885880, -59.5324249, 58.9563217
4: -23.9724770, 28.7407074, -22.9012489, 27.5461750, -51.5186539, 51.6419563

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8692083, upper bound: 17.8079590
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8692083, upper bound: 17.8336619
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.5872440, 7.9929218, -9.0013008, 7.5648065, -17.1520500, 16.9942226
1: -37.5564880, 30.9611835, -35.3616943, 29.4438133, -67.0002975, 66.3228760
2: -18.3394547, 28.1553612, -17.1289692, 26.5229263, -44.8623810, 45.2843323
3: -32.7510910, 28.0313568, -30.8644867, 26.5948696, -59.3459625, 58.8958435
4: -24.1285267, 29.0189381, -22.6808529, 27.3175602, -51.4460869, 51.6997871

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8239343, upper bound: 17.8367812
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8239443, upper bound: 17.8130413
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.5109835, 7.9441323, -10.3218880, 8.5799026, -18.0908852, 18.2660198
1: -37.3659744, 30.7502995, -40.5616798, 33.1231194, -70.4890747, 71.3119812
2: -18.1058884, 27.9925270, -19.6370163, 30.3212185, -48.4271011, 47.6295433
3: -32.6438370, 27.8008327, -35.4595146, 29.9217396, -62.5655746, 63.2603416
4: -23.9724770, 28.7407074, -26.0334129, 30.9995728, -54.9720497, 54.7741203

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026830, upper bound: 17.9027093
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.9026830, upper bound: 17.9027093
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.5872440, 7.9929218, -10.0871792, 8.3898401, -17.9770832, 18.0801010
1: -37.5564880, 30.9611835, -39.6141396, 32.3487854, -69.9052734, 70.5753250
2: -18.3394547, 28.1553612, -19.1885662, 29.6788445, -48.0182991, 47.3439217
3: -32.7510910, 28.0313568, -34.6299553, 29.2403488, -61.9914398, 62.6613083
4: -24.1285267, 29.0189381, -25.4263000, 30.3402138, -54.4687386, 54.4452286

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8260189, upper bound: 17.8787833
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8128930, upper bound: 17.8242865
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.73 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8062908, upper bound: 17.8692083
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8062908, upper bound: 17.8692083
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8145100, upper bound: 17.8767008
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8130413, upper bound: 17.8239444
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8692083, upper bound: 17.8079590
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8692083, upper bound: 17.8336619
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8239343, upper bound: 17.8367812
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8239443, upper bound: 17.8130413
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.9026830, upper bound: 17.9027093
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.9026830, upper bound: 17.9027093
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8260189, upper bound: 17.8787833
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 0, lower bound: -17.8128930, upper bound: 17.8242865

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.2072668, 6.9692569, -9.5109835, 7.9441323, -16.1513996, 16.4802399
1: -32.2548981, 27.3039436, -37.3659744, 30.7502995, -63.0051956, 64.6698990
2: -15.6097517, 24.1825752, -18.1058884, 27.9925270, -43.6022797, 42.2884636
3: -28.1271381, 24.6478672, -32.6438370, 27.8008327, -55.9279671, 57.2917023
4: -20.6859665, 25.1434441, -23.9724770, 28.7407074, -49.4266739, 49.1159172

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7711373, upper bound: 17.8577937
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7998198, upper bound: 17.8669068
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.2072668, 6.9692569, -9.5872440, 7.9929218, -16.2001839, 16.5564995
1: -32.2548981, 27.3039436, -37.5564880, 30.9611835, -63.2160797, 64.8604279
2: -15.6097517, 24.1825752, -18.3394547, 28.1553612, -43.7651100, 42.5220299
3: -28.1271381, 24.6478672, -32.7510910, 28.0313568, -56.1584930, 57.3989563
4: -20.6859665, 25.1434441, -24.1285267, 29.0189381, -49.7048988, 49.2719612

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7711373, upper bound: 17.8577938
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7998198, upper bound: 17.8669068
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6899605, 7.3238630, -10.0071678, 8.3412704, -17.0312290, 17.3310318
1: -34.0813217, 28.6633034, -39.3127899, 32.2082672, -66.2895889, 67.9760895
2: -16.6928825, 25.5631542, -18.9993477, 29.3747463, -46.0676193, 44.5624886
3: -29.6711578, 25.9272728, -34.3978424, 29.0799122, -58.7510681, 60.3251114
4: -21.8786697, 26.6128159, -25.2551899, 30.1101170, -51.9887848, 51.8679924

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8111390, upper bound: 17.8739253
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7950171, upper bound: 17.8725721
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8058963, upper bound: 17.8743406
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.5109835, 7.9441323, -8.2072668, 6.9692569, -16.4802399, 16.1513977
1: -37.3659744, 30.7502995, -32.2548981, 27.3039436, -64.6698990, 63.0051956
2: -18.1058884, 27.9925270, -15.6097517, 24.1825752, -42.2884636, 43.6022797
3: -32.6438370, 27.8008327, -28.1271381, 24.6478672, -57.2917023, 55.9279671
4: -23.9724770, 28.7407074, -20.6859665, 25.1434441, -49.1159172, 49.4266701

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8699002, upper bound: 17.8027546
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8337009, upper bound: 17.8016279
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.5109835, 7.9441323, -8.7493315, 7.3718662, -16.8828506, 16.6934624
1: -37.3659744, 30.7502995, -34.3156853, 28.8483829, -66.2143402, 65.0659866
2: -18.1058884, 27.9925270, -16.8087482, 25.7330456, -43.8389359, 44.8012695
3: -32.6438370, 27.8008327, -29.8720760, 26.0954170, -58.7392540, 57.6729050
4: -23.9724770, 28.7407074, -22.0263462, 26.7866917, -50.7591705, 50.7670517

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8228534, upper bound: 17.8119690
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8677760, upper bound: 17.7998290
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.5109835, 7.9441323, -9.5109835, 7.9441323, -17.4551163, 17.4551163
1: -37.3659744, 30.7502995, -37.3659744, 30.7502995, -68.1162643, 68.1162643
2: -18.1058884, 27.9925270, -18.1058884, 27.9925270, -46.0984154, 46.0984154
3: -32.6438370, 27.8008327, -32.6438370, 27.8008327, -60.4446716, 60.4446716
4: -23.9724770, 28.7407074, -23.9724770, 28.7407074, -52.7131844, 52.7131844

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8855654, upper bound: 17.8260419
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8352908, upper bound: 17.8244370
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.5109835, 7.9441323, -9.5872440, 7.9929218, -17.5039062, 17.5313740
1: -37.3659744, 30.7502995, -37.5564880, 30.9611835, -68.3271561, 68.3067856
2: -18.1058884, 27.9925270, -18.3394547, 28.1553612, -46.2612495, 46.3319817
3: -32.6438370, 27.8008327, -32.7510910, 28.0313568, -60.6751938, 60.5519257
4: -23.9724770, 28.7407074, -24.1285267, 29.0189381, -52.9914131, 52.8692322

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8855653, upper bound: 17.8260419
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8352908, upper bound: 17.8244370
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.5270128, 7.9461823, -10.0071678, 8.3412704, -17.8682823, 17.9533501
1: -37.3180542, 30.7838192, -39.3127899, 32.2082672, -69.5263062, 70.0966034
2: -18.2267265, 27.9892673, -18.9993477, 29.3747463, -47.6014709, 46.9886169
3: -32.5461769, 27.8701172, -34.3978424, 29.0799122, -61.6260910, 62.2679558
4: -23.9785881, 28.8482628, -25.2551899, 30.1101170, -54.0886993, 54.1034431

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8087344, upper bound: 17.8712262
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8214124, upper bound: 17.8771422
time: 0.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.84 seconds
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.7711373, upper bound: 17.8577937
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.7998198, upper bound: 17.8669068
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.7711373, upper bound: 17.8577938
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.7998198, upper bound: 17.8669068
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.7950171, upper bound: 17.8725721
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8058963, upper bound: 17.8743406
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8699002, upper bound: 17.8027546
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8337009, upper bound: 17.8016279
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8228534, upper bound: 17.8119690
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8677760, upper bound: 17.7998290
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8855654, upper bound: 17.8260419
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8352908, upper bound: 17.8244370
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8855653, upper bound: 17.8260419
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8352908, upper bound: 17.8244370
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8087344, upper bound: 17.8712262
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -17.8214124, upper bound: 17.8771422

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.4676991, 6.4021091, -9.4171934, 7.8709803, -15.3386793, 15.8193026
1: -29.3328896, 25.2166233, -36.9963608, 30.4787102, -59.8115997, 62.2129822
2: -14.2339802, 22.0512600, -17.9336376, 27.7205009, -41.9544792, 39.9848976
3: -25.5683327, 22.7662430, -32.3215370, 27.5569630, -53.1252899, 55.0877800
4: -18.8227520, 23.1316185, -23.7364922, 28.4783077, -47.3010559, 46.8681107

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7817959, upper bound: 17.8270214
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7817959, upper bound: 17.8677760
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.4676991, 6.4021091, -9.4798775, 7.9084549, -15.3761539, 15.8819866
1: -29.3328896, 25.2166233, -37.1336555, 30.6517677, -59.9846573, 62.3502808
2: -14.2339802, 22.0512600, -18.1368523, 27.8402386, -42.0742188, 40.1881104
3: -25.5683327, 22.7662430, -32.3810921, 27.7545567, -53.3228836, 55.1473351
4: -18.8227520, 23.1316185, -23.8573227, 28.7185936, -47.5413322, 46.9889297

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7817959, upper bound: 17.8270214
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7872161, upper bound: 17.8669068
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.8803701, 6.7357445, -9.2983828, 7.7870984, -15.6674662, 16.0341263
1: -30.8293400, 26.6018944, -36.5123787, 30.1613903, -60.9907188, 63.1142731
2: -15.3275795, 23.3810558, -17.6477585, 27.3667030, -42.6942825, 41.0288124
3: -26.6872749, 24.1738052, -31.9047699, 27.2678242, -53.9550819, 56.0785713
4: -19.7472057, 24.6753139, -23.4325504, 28.1749058, -47.9221115, 48.1078644

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8630638
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8725721
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.8519564, 6.6851311, -9.9014320, 8.2577028, -16.1096573, 16.5865593
1: -30.7409973, 26.3228207, -38.8975296, 31.9046249, -62.6456146, 65.2203522
2: -15.1209888, 23.1824665, -18.8008804, 29.0648270, -44.1858139, 41.9833374
3: -26.7309647, 23.8386784, -34.0318604, 28.8067646, -55.5377274, 57.8705368
4: -19.7420025, 24.3633404, -24.9894428, 29.8128510, -49.5548515, 49.3527794

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8056814, upper bound: 17.8709056
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8601294
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8743405
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.4517412, 7.9154038, -8.1597357, 6.9314404, -16.3831806, 16.0751400
1: -37.1512413, 30.6756077, -32.0669060, 27.1577129, -64.3089523, 62.7425079
2: -17.9562969, 27.7387924, -15.5202303, 24.0493240, -42.0056190, 43.2590218
3: -32.4846649, 27.7031937, -27.9643879, 24.5160751, -57.0007362, 55.6675797
4: -23.8567657, 28.5601749, -20.5670242, 25.0067616, -48.8635254, 49.1271973

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8590348, upper bound: 17.7661173
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8694851, upper bound: 17.8012722
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8370109, upper bound: 17.7403802
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8093719, 7.3991070, -8.6362190, 7.2864060, -16.0957775, 16.0353260
1: -34.6173935, 28.7227898, -33.8676910, 28.5331879, -63.1505699, 62.5904808
2: -16.8223629, 25.9580612, -16.5993385, 25.4086628, -42.2310181, 42.5573959
3: -30.2329674, 25.9846096, -29.4779072, 25.8153019, -56.0482712, 55.4625130
4: -22.2092323, 26.7780247, -21.7394676, 26.4846134, -48.6938477, 48.5174942

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8708357, upper bound: 17.8249770
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8693370, upper bound: 17.8002160
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.4517412, 7.9154038, -9.4614429, 7.9048338, -17.3565750, 17.3768463
1: -37.1512413, 30.6756077, -37.1703110, 30.6005535, -67.7517853, 67.8459091
2: -17.9562969, 27.7387924, -18.0124798, 27.8525505, -45.8088379, 45.7512741
3: -32.4846649, 27.7031937, -32.4754333, 27.6654053, -60.1500702, 60.1786270
4: -23.8567657, 28.5601749, -23.8488083, 28.5986786, -52.4554443, 52.4089737

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8804135, upper bound: 17.8102734
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8940580, upper bound: 17.8345481
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.4517412, 7.9154038, -9.5270128, 7.9461823, -17.3979225, 17.4424171
1: -37.1512413, 30.6756077, -37.3180542, 30.7838192, -67.9350586, 67.9936447
2: -17.9562969, 27.7387924, -18.2267265, 27.9892673, -45.9455643, 45.9655190
3: -32.4846649, 27.7031937, -32.5461769, 27.8701172, -60.3547745, 60.2493706
4: -23.8567657, 28.5601749, -23.9785881, 28.8482628, -52.7050247, 52.5387611

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8805219, upper bound: 17.8092034
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8832927, upper bound: 17.8214532
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.6021452, 7.2457342, -9.2983828, 7.7870984, -16.3892422, 16.5441170
1: -33.6806526, 28.3082275, -36.5123787, 30.1613903, -63.8420410, 64.8206024
2: -16.5842266, 25.3920174, -17.6477585, 27.3667030, -43.9509277, 43.0397682
3: -29.2485256, 25.7472992, -31.9047699, 27.2678242, -56.5163345, 57.6520691
4: -21.5821838, 26.4831047, -23.4325504, 28.1749058, -49.7570877, 49.9156570

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085760, upper bound: 17.8689645
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8080562, upper bound: 17.8560803
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.7402859, 7.3273911, -9.9014320, 8.2577028, -16.9979897, 17.2288227
1: -34.2258873, 28.5099182, -38.8975296, 31.9046249, -66.1305008, 67.4074478
2: -16.7329445, 25.6812382, -18.8008804, 29.0648270, -45.7977715, 44.4821091
3: -29.8350601, 25.8272972, -34.0318604, 28.8067646, -58.6418152, 59.8591576
4: -21.9899235, 26.6435699, -24.9894428, 29.8128510, -51.8027725, 51.6330070

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.53 seconds
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.7817959, upper bound: 17.8270214
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.7817959, upper bound: 17.8677760
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.7817959, upper bound: 17.8270214
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.7872161, upper bound: 17.8669068
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8630638
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8725721
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8601294
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8743405
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8694851, upper bound: 17.8012722
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8370109, upper bound: 17.7403802
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8708357, upper bound: 17.8249770
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8693370, upper bound: 17.8002160
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8804135, upper bound: 17.8102734
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8940580, upper bound: 17.8345481
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8805219, upper bound: 17.8092034
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8832927, upper bound: 17.8214532
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8085760, upper bound: 17.8689645
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8080562, upper bound: 17.8560803
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.4676991, 6.4021091, -8.8093719, 7.3991070, -14.8668051, 15.2114811
1: -29.3328896, 25.2166233, -34.6173935, 28.7227898, -58.0556793, 59.8340149
2: -14.2339802, 22.0512600, -16.8223629, 25.9580612, -40.1920357, 38.8736229
3: -25.5683327, 22.7662430, -30.2329674, 25.9846096, -51.5529404, 52.9992104
4: -18.8227520, 23.1316185, -22.2092323, 26.7780247, -45.6007729, 45.3408508

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7379900, upper bound: 17.8209109
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7803973, upper bound: 17.8584956
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.4676991, 6.4021091, -8.7999458, 7.3732409, -14.8409405, 15.2020550
1: -29.3328896, 25.2166233, -34.4601173, 28.6844330, -58.0173225, 59.6767426
2: -14.2339802, 22.0512600, -16.8442154, 25.8455009, -40.0794792, 38.8954735
3: -25.5683327, 22.7662430, -30.0377827, 25.9852562, -51.5535889, 52.8040237
4: -18.8227520, 23.1316185, -22.1389160, 26.8108521, -45.6336060, 45.2705307

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7369907, upper bound: 17.8006076
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7859698, upper bound: 17.8635963
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.8803701, 6.7357445, -8.6580315, 7.2963896, -15.1767597, 15.3937759
1: -30.8293400, 26.6018944, -34.0170441, 28.4055386, -59.2348709, 60.6189384
2: -15.3275795, 23.3810558, -16.4303532, 25.4585686, -40.7861443, 39.8114014
3: -26.6872749, 24.1738052, -29.7197647, 25.6733513, -52.3606262, 53.8935699
4: -19.7472057, 24.6753139, -21.8356495, 26.3721561, -46.1193619, 46.5109596

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8204222
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8630638
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.8803701, 6.7357445, -8.9315987, 7.4924607, -15.3728304, 15.6673431
1: -30.8293400, 26.6018944, -34.9783516, 29.1649990, -59.9943390, 61.5802460
2: -15.3275795, 23.3810558, -17.0554295, 26.1942902, -41.5218697, 40.4364853
3: -26.6872749, 24.1738052, -30.4928913, 26.4158897, -53.1031647, 54.6666832
4: -19.7472057, 24.6753139, -22.4788132, 27.2196369, -46.9668427, 47.1541252

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8725721
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8725720
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.8519564, 6.6851311, -8.4854431, 7.1639380, -15.0158939, 15.1705742
1: -30.7409973, 26.3228207, -33.3543930, 27.9147034, -58.6556969, 59.6772156
2: -15.1209888, 23.1824665, -16.1281109, 25.0088215, -40.1298103, 39.3105774
3: -26.7309647, 23.8386784, -29.0962524, 25.2967300, -52.0276947, 52.9349289
4: -19.7420025, 24.3633404, -21.3634624, 25.9582558, -45.7002487, 45.7268028

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8601294
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8601294
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.8519564, 6.6851311, -9.2663279, 7.7600579, -15.6120138, 15.9514561
1: -30.7409973, 26.3228207, -36.4070740, 30.0852451, -60.8262405, 62.7298927
2: -15.1209888, 23.1824665, -17.6246490, 27.2220001, -42.3429871, 40.8071098
3: -26.7309647, 23.8386784, -31.8322182, 27.1748791, -53.9058456, 55.6708870
4: -19.7420025, 24.3633404, -23.3914528, 28.0394897, -47.7814903, 47.7547798

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8692641
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8692641
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.4162989, 7.8881068, -7.8663883, 6.7064772, -16.1227741, 15.7544956
1: -37.0100212, 30.5759258, -30.8994789, 26.3289680, -63.3389893, 61.4754028
2: -17.8874588, 27.6389809, -14.9488449, 23.2205372, -41.1079865, 42.5878258
3: -32.3587227, 27.6134033, -26.9385128, 23.7713203, -56.1300392, 54.5519180
4: -23.7655506, 28.4627743, -19.8190651, 24.2090321, -47.9745827, 48.2818375

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8561619, upper bound: 17.7829507
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.3437672, 7.0262022, -8.0148029, 6.8487353, -15.1925020, 15.0410051
1: -32.7600784, 27.2942581, -31.4167747, 27.0508690, -59.8109474, 58.7110291
2: -15.9520369, 24.6677799, -15.4422750, 23.7034855, -39.6555214, 40.1100464
3: -28.6095295, 24.7034817, -27.2535534, 24.5060043, -53.1155319, 51.9570351
4: -21.0216656, 25.4773350, -20.1291656, 25.0205460, -46.0422134, 45.6064987

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8483014, upper bound: 17.8003222
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8707175, upper bound: 17.8247608
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.7837048, 7.3793225, -8.3006105, 7.0327530, -15.8164577, 15.6799335
1: -34.5158997, 28.6481972, -32.5227051, 27.5853767, -62.1012764, 61.1709023
2: -16.7739277, 25.8856144, -15.9755430, 24.4824867, -41.2564087, 41.8611565
3: -30.1439285, 25.9177475, -28.2964039, 24.9647579, -55.1086884, 54.2141495
4: -22.1438522, 26.7069530, -20.8857975, 25.5850964, -47.7289505, 47.5927429

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8482903, upper bound: 17.7892178
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8692686, upper bound: 17.7996170
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6580315, 7.2963896, -7.6948395, 6.5498452, -15.2078762, 14.9912291
1: -34.0170441, 28.4055386, -30.2609119, 25.6755867, -59.6926308, 58.6664505
2: -16.4303532, 25.4585686, -14.6454887, 22.6898785, -39.1202316, 40.1040573
3: -29.7197647, 25.6733513, -26.3800907, 23.2771301, -52.9968948, 52.0534439
4: -21.8356495, 26.3721561, -19.3823891, 23.7483749, -45.5840225, 45.7545433

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8226364, upper bound: 17.8088369
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8226364, upper bound: 17.8102734
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.3605661, 7.8441429, -8.7616396, 7.3608155, -16.7213802, 16.6057816
1: -36.7916641, 30.4124203, -34.4289017, 28.5772190, -65.3688812, 64.8413239
2: -17.7890244, 27.4719048, -16.7315140, 25.8217411, -43.6107521, 44.2034187
3: -32.1708183, 27.4663143, -30.0683460, 25.8531494, -58.0239677, 57.5346527
4: -23.6280575, 28.3028297, -22.0887489, 26.6399879, -50.2680435, 50.3915787

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8227418, upper bound: 17.8217120
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8227418, upper bound: 17.8345481
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6580315, 7.2963896, -8.6021452, 7.2457342, -15.9037657, 15.8985348
1: -34.0170441, 28.4055386, -33.6806526, 28.3082275, -62.3252678, 62.0861893
2: -16.4303532, 25.4585686, -16.5842266, 25.3920174, -41.8223610, 42.0427895
3: -29.7197647, 25.6733513, -29.2485256, 25.7472992, -55.4670639, 54.9218750
4: -21.8356495, 26.3721561, -21.5821838, 26.4831047, -48.3187485, 47.9543381

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8171841, upper bound: 17.8075165
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8226029, upper bound: 17.8092034
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.3605661, 7.8441429, -8.7402859, 7.3273911, -16.6879578, 16.5844288
1: -36.7916641, 30.4124203, -34.2258873, 28.5099182, -65.3015823, 64.6383057
2: -17.7890244, 27.4719048, -16.7329445, 25.6812382, -43.4702606, 44.2048492
3: -32.1708183, 27.4663143, -29.8350601, 25.8272972, -57.9981155, 57.3013687
4: -23.6280575, 28.3028297, -21.9899235, 26.6435699, -50.2716255, 50.2927551

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8226205, upper bound: 17.8123853
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8226205, upper bound: 17.8214532
time: 3.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.1786079, 6.9158902, -8.0261688, 6.8345747, -15.0131826, 14.9420547
1: -31.9874687, 27.0390587, -31.4366665, 26.7455196, -58.7329865, 58.4757195
2: -15.7956524, 24.2188644, -15.2999859, 23.7203999, -39.5160408, 39.5188370
3: -27.7718067, 24.6017704, -27.4118500, 24.1750565, -51.9468613, 52.0136108
4: -20.5080299, 25.2984352, -20.2305450, 24.7487106, -45.2567368, 45.5289803

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085760, upper bound: 17.8689645
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085759, upper bound: 17.8689644
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.7402859, 7.3273911, -9.3594456, 7.8430247, -16.5833111, 16.6868362
1: -34.2258873, 28.5099182, -36.7872276, 30.4083672, -64.6342545, 65.2971420
2: -16.7329445, 25.6812382, -17.7867584, 27.4684029, -44.2013474, 43.4679947
3: -29.8350601, 25.8272972, -32.1667938, 27.4625969, -57.2976570, 57.9940910
4: -21.9899235, 26.6435699, -23.6250992, 28.2990532, -50.2889786, 50.2686691

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8083842, upper bound: 17.8378156
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206975, upper bound: 17.8611016
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.7402859, 7.3273911, -9.4655151, 7.9042768, -16.6445618, 16.7929058
1: -34.2258873, 28.5099182, -37.0961876, 30.6781178, -64.9040070, 65.6061096
2: -16.7329445, 25.6812382, -18.0687485, 27.7018509, -44.4347954, 43.7499847
3: -29.8350601, 25.8272972, -32.3743858, 27.7506046, -57.5856628, 58.2016830
4: -21.9899235, 26.6435699, -23.8506012, 28.6616802, -50.6516037, 50.4941711

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8083842, upper bound: 17.8426854
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.04 seconds
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7379900, upper bound: 17.8209109
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7803973, upper bound: 17.8584956
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7369907, upper bound: 17.8006076
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7859698, upper bound: 17.8635963
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8204222
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8630638
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8725721
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8725720
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8601294
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8601294
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8692641
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8692641
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8483014, upper bound: 17.8003222
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8707175, upper bound: 17.8247608
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8482903, upper bound: 17.7892178
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8692686, upper bound: 17.7996170
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8226364, upper bound: 17.8088369
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8226364, upper bound: 17.8102734
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8227418, upper bound: 17.8217120
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8227418, upper bound: 17.8345481
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8171841, upper bound: 17.8075165
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8226029, upper bound: 17.8092034
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8226205, upper bound: 17.8123853
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8226205, upper bound: 17.8214532
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8085760, upper bound: 17.8689645
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8085759, upper bound: 17.8689644
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8083842, upper bound: 17.8378156
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8206975, upper bound: 17.8611016
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -17.8206979, upper bound: 17.8622303

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.2083950, 6.1951623, -8.7010431, 7.2930741, -14.5014687, 14.8962021
1: -28.3006630, 24.4443398, -34.0669899, 28.3825035, -56.6831665, 58.5113297
2: -13.7494001, 21.2990627, -16.6593685, 25.5630665, -39.3124657, 37.9584198
3: -24.6659660, 22.0737915, -29.6916275, 25.7167187, -50.3826828, 51.7654190
4: -18.1683788, 22.3953533, -21.8862648, 26.5258293, -44.6942062, 44.2816162

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7878322, upper bound: 17.8485060
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7884229, upper bound: 17.8581970
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.5781627, 6.5181785, -8.6580315, 7.2963896, -14.8745518, 15.1762094
1: -29.6119995, 25.8476696, -34.0170441, 28.4055386, -58.0175362, 59.8647156
2: -14.7513695, 22.5336590, -16.4303532, 25.4585686, -40.2099342, 38.9640121
3: -25.6100578, 23.4752998, -29.7197647, 25.6733513, -51.2834091, 53.1950645
4: -18.9980183, 23.8903866, -21.8356495, 26.3721561, -45.3701744, 45.7260361

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7934659, upper bound: 17.8628632
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.7469807, 6.6542716, -8.9315987, 7.4924607, -15.2394409, 15.5858688
1: -30.3022652, 26.3353329, -34.9783516, 29.1649990, -59.4672623, 61.3136787
2: -15.0589609, 22.9798450, -17.0554295, 26.1942902, -41.2532501, 40.0352745
3: -26.2437019, 23.9160156, -30.4928913, 26.4158897, -52.6595917, 54.4089012
4: -19.4321709, 24.3534050, -22.4788132, 27.2196369, -46.6518097, 46.8322182

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8635938
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8725720
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.5781627, 6.5181785, -8.9315987, 7.4924607, -15.0706234, 15.4497757
1: -29.6119995, 25.8476696, -34.9783516, 29.1649990, -58.7770004, 60.8260193
2: -14.7513695, 22.5336590, -17.0554295, 26.1942902, -40.9456596, 39.5890884
3: -25.6100578, 23.4752998, -30.4928913, 26.4158897, -52.0259476, 53.9681778
4: -18.9980183, 23.8903866, -22.4788132, 27.2196369, -46.2176552, 46.3691978

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8635938
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8725720
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.7390552, 6.6119957, -8.4854431, 7.1639380, -14.9029932, 15.0974388
1: -30.2898140, 26.0954514, -33.3543930, 27.9147034, -58.2045097, 59.4498291
2: -14.8736029, 22.8132610, -16.1281109, 25.0088215, -39.8824234, 38.9413719
3: -26.3403473, 23.6168098, -29.0962524, 25.2967300, -51.6370773, 52.7130623
4: -19.4622726, 24.0755539, -21.3634624, 25.9582558, -45.4205284, 45.4390182

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8007777, upper bound: 17.8600942
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8569230
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.5026493, 6.4323788, -8.4854431, 7.1639380, -14.6665878, 14.9178219
1: -29.3362160, 25.4468307, -33.3543930, 27.9147034, -57.2509079, 58.8012238
2: -14.4507380, 22.2111225, -16.1281109, 25.0088215, -39.4595566, 38.3392334
3: -25.4694633, 23.0382729, -29.0962524, 25.2967300, -50.7661934, 52.1345253
4: -18.8570747, 23.4657974, -21.3634624, 25.9582558, -44.8153229, 44.8292618

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8007777, upper bound: 17.8600943
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8581283
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.7390552, 6.6119957, -9.2663279, 7.7600579, -15.4991131, 15.8783188
1: -30.2898140, 26.0954514, -36.4070740, 30.0852451, -60.3750572, 62.5025139
2: -14.8736029, 22.8132610, -17.6246490, 27.2220001, -42.0956039, 40.4379120
3: -26.3403473, 23.6168098, -31.8322182, 27.1748791, -53.5152206, 55.4490280
4: -19.4622726, 24.0755539, -23.3914528, 28.0394897, -47.5017624, 47.4669952

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8630315
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8058236, upper bound: 17.8692641
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.5026493, 6.4323788, -9.2663279, 7.7600579, -15.2627068, 15.6987066
1: -29.3362160, 25.4468307, -36.4070740, 30.0852451, -59.4214516, 61.8539009
2: -14.4507380, 22.2111225, -17.6246490, 27.2220001, -41.6727371, 39.8357697
3: -25.4694633, 23.0382729, -31.8322182, 27.1748791, -52.6443405, 54.8704834
4: -18.8570747, 23.4657974, -23.3914528, 28.0394897, -46.8965645, 46.8572502

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8630315
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8058236, upper bound: 17.8692641
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.0587130, 6.7997308, -7.9248581, 6.7735925, -14.8323059, 14.7245884
1: -31.6290512, 26.4341450, -31.0591202, 26.7624054, -58.3914490, 57.4932632
2: -15.4185276, 23.8419552, -15.2735634, 23.4459114, -38.8644409, 39.1155167
3: -27.6192970, 23.9329758, -26.9410877, 24.2506790, -51.8699722, 50.8740616
4: -20.2993298, 24.6619148, -19.9001160, 24.7565556, -45.0558739, 44.5620270

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8624030, upper bound: 17.8065475
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8624030, upper bound: 17.8247608
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.4920979, 7.1477027, -8.2017021, 6.9535532, -15.4456482, 15.3494053
1: -33.3568344, 27.7709141, -32.1272812, 27.2854538, -60.6422882, 59.8981934
2: -16.2290630, 25.0428391, -15.7927504, 24.2045670, -40.4336319, 40.8355827
3: -29.1290913, 25.1316872, -27.9492798, 24.6994476, -53.8285370, 53.0809669
4: -21.4046745, 25.8735676, -20.6340675, 25.3075962, -46.7122726, 46.5076370

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7786589
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7996170
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.1661940, 6.9140811, -8.0261688, 6.8345747, -15.0007677, 14.9402504
1: -31.9513187, 27.0860863, -31.4366665, 26.7455196, -58.6968384, 58.5227509
2: -15.7361317, 24.1032448, -15.2999859, 23.7203999, -39.4565277, 39.4032288
3: -27.7711315, 24.6224098, -27.4118500, 24.1750565, -51.9461899, 52.0342484
4: -20.5043335, 25.2644825, -20.2305450, 24.7487106, -45.2530441, 45.4950218

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8665161
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8689645
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.8005185, 6.6397552, -8.0261688, 6.8345747, -14.6350937, 14.6659231
1: -30.4766083, 26.0748787, -31.4366665, 26.7455196, -57.2221298, 57.5115356
2: -15.0684252, 23.1567383, -15.2999859, 23.7203999, -38.7888260, 38.4567223
3: -26.4369164, 23.7238541, -27.4118500, 24.1750565, -50.6119728, 51.1357002
4: -19.5588493, 24.3224087, -20.2305450, 24.7487106, -44.3075562, 44.5529404

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8665161
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8689645
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.4281263, 7.0879607, -9.3353586, 7.8244047, -16.2525311, 16.4233189
1: -32.9812317, 27.6122894, -36.6918869, 30.3379402, -63.3191719, 64.3041763
2: -16.1435299, 24.8028431, -17.7415390, 27.4003506, -43.5438690, 42.5443764
3: -28.7412357, 25.0208511, -32.0833473, 27.3989162, -56.1401520, 57.1041985
4: -21.1967201, 25.7855167, -23.5639305, 28.2320309, -49.4287491, 49.3494415

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206974, upper bound: 17.8611014
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206467, upper bound: 17.8610999
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.8027897, 7.3821054, -9.4655151, 7.9042768, -16.7070637, 16.8476200
1: -34.4857254, 28.7678318, -37.0961876, 30.6781178, -65.1638412, 65.8640213
2: -16.8040066, 25.7537098, -18.0687485, 27.7018509, -44.5058556, 43.8224564
3: -30.0783405, 26.0442696, -32.3743858, 27.7506046, -57.8289452, 58.4186516
4: -22.1723690, 26.8013783, -23.8506012, 28.6616802, -50.8340416, 50.6519775

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8469639
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8565074
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.3598843, 7.0487337, -9.4655151, 7.9042768, -16.2641602, 16.5142460
1: -32.7005844, 27.5349121, -37.0961876, 30.6781178, -63.3787003, 64.6310959
2: -16.0005322, 24.5990105, -18.0687485, 27.7018509, -43.7023849, 42.6677589
3: -28.4687157, 24.9388924, -32.3743858, 27.7506046, -56.2193146, 57.3132744
4: -21.0292091, 25.6482048, -23.8506012, 28.6616802, -49.6908798, 49.4988060

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8469640
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8565074
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.66 seconds
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.7878322, upper bound: 17.8485060
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.7884229, upper bound: 17.8581970
NS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8635938
NS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8725720
NS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8635938
NS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8725720
NS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8007777, upper bound: 17.8600942
NS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8569230
NS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8007777, upper bound: 17.8600943
NS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8012356, upper bound: 17.8581283
NS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8630315
NS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8058236, upper bound: 17.8692641
NS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8630315
NS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8058236, upper bound: 17.8692641
NS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8624030, upper bound: 17.8065475
NS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8624030, upper bound: 17.8247608
NS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7786589
NS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7996170
NS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8665161
NS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8689645
NS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8665161
NS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8085139, upper bound: 17.8689645
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8206974, upper bound: 17.8611014
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8206467, upper bound: 17.8610999
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8469639
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8565074
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8469640
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -17.8152299, upper bound: 17.8565074

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.7469807, 6.6542716, -8.6134920, 7.2626133, -15.0095940, 15.2677622
1: -30.3022652, 26.3353329, -33.7376022, 28.4294949, -58.7317543, 60.0729332
2: -15.0589609, 22.9798450, -16.5684509, 25.3417053, -40.4006577, 39.5482941
3: -26.2437019, 23.9160156, -29.3290005, 25.8280106, -52.0717125, 53.2450180
4: -19.4321709, 24.3534050, -21.6396294, 26.5144291, -45.9465942, 45.9930344

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.7469807, 6.6542716, -8.7983608, 7.3789415, -15.1259222, 15.4526320
1: -30.3022652, 26.3353329, -34.4680099, 28.7565403, -59.0588036, 60.8033409
2: -15.0589609, 22.9798450, -16.7960987, 25.7419415, -40.8008995, 39.7759399
3: -26.2437019, 23.9160156, -30.0620613, 26.0341454, -52.2778435, 53.9780769
4: -19.4321709, 24.3534050, -22.1607056, 26.7901802, -46.2223511, 46.5141106

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.5781627, 6.5181785, -8.6134920, 7.2626133, -14.8407764, 15.1316690
1: -29.6119995, 25.8476696, -33.7376022, 28.4294949, -58.0414848, 59.5852737
2: -14.7513695, 22.5336590, -16.5684509, 25.3417053, -40.0930710, 39.1021118
3: -25.6100578, 23.4752998, -29.3290005, 25.8280106, -51.4380684, 52.8042946
4: -18.9980183, 23.8903866, -21.6396294, 26.5144291, -45.5124435, 45.5300140

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7750767, upper bound: 17.8362228
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8635938
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.5781627, 6.5181785, -8.7983608, 7.3789415, -14.9571037, 15.3165398
1: -29.6119995, 25.8476696, -34.4680099, 28.7565403, -58.3685379, 60.3156815
2: -14.7513695, 22.5336590, -16.7960987, 25.7419415, -40.4933090, 39.3297577
3: -25.6100578, 23.4752998, -30.0620613, 26.0341454, -51.6442032, 53.5373535
4: -18.9980183, 23.8903866, -22.1607056, 26.7901802, -45.7882004, 46.0510941

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7750766, upper bound: 17.8453178
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8679904
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.3074198, 6.2714930, -7.3053565, 6.2432756, -13.5506935, 13.5768490
1: -28.5873909, 24.7526608, -28.6836700, 24.3827972, -52.9701881, 53.4363327
2: -14.0708933, 21.6166821, -13.9342489, 21.6475124, -35.7184067, 35.5509262
3: -24.8636036, 22.4055328, -25.0007229, 22.0809174, -46.9445190, 47.4062576
4: -18.3746891, 22.8368244, -18.3761883, 22.5888424, -40.9635315, 41.2130127

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8117068, upper bound: 17.8573185
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8217249, upper bound: 17.8686260
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.1117411, 6.1235600, -7.3053565, 6.2432756, -13.3550158, 13.4289169
1: -27.8034840, 24.2234879, -28.6836700, 24.3827972, -52.1862793, 52.9071503
2: -13.7204676, 21.1240349, -13.9342489, 21.6475124, -35.3679733, 35.0582848
3: -24.1356583, 21.9334717, -25.0007229, 22.0809174, -46.2165756, 46.9341965
4: -17.8755646, 22.3398705, -18.3761883, 22.5888424, -40.4644089, 40.7160530

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3074198, 6.2714930, -8.1325941, 6.8638730, -14.1712933, 14.4040871
1: -28.5873909, 24.7526608, -31.9281921, 26.6394501, -55.2268410, 56.6808548
2: -14.0708933, 21.6166821, -15.5028181, 23.9614868, -38.0323792, 37.1194992
3: -24.8636036, 22.4055328, -27.9155197, 24.0500088, -48.9136124, 50.3210526
4: -18.3746891, 22.8368244, -20.5240555, 24.7739277, -43.1486092, 43.3608780

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8318722, upper bound: 17.8790051
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8332327, upper bound: 17.8792645
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.7390552, 6.6119957, -9.0996466, 7.6320057, -15.3710613, 15.7116423
1: -30.2898140, 26.0954514, -35.7427864, 29.5985088, -59.8883095, 61.8382301
2: -14.8736029, 22.8132610, -17.3090534, 26.7460995, -41.6197014, 40.1223145
3: -26.3403473, 23.6168098, -31.2481956, 26.7394676, -53.0798149, 54.8650017
4: -19.4622726, 24.0755539, -22.9685173, 27.5716896, -47.0339622, 47.0440712

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8315257, upper bound: 17.8713028
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8328246, upper bound: 17.8697862
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.1117411, 6.1235600, -8.1325941, 6.8638730, -13.9756145, 14.2561541
1: -27.8034840, 24.2234879, -31.9281921, 26.6394501, -54.4429321, 56.1516800
2: -13.7204676, 21.1240349, -15.5028181, 23.9614868, -37.6819534, 36.6268539
3: -24.1356583, 21.9334717, -27.9155197, 24.0500088, -48.1856689, 49.8489876
4: -17.8755646, 22.3398705, -20.5240555, 24.7739277, -42.6494904, 42.8639183

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8630316
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8629350
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8024683, upper bound: 17.8630315
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.5026493, 6.4323788, -9.0996466, 7.6320057, -15.1346550, 15.5320253
1: -29.3362160, 25.4468307, -35.7427864, 29.5985088, -58.9347076, 61.1896172
2: -14.4507380, 22.2111225, -17.3090534, 26.7460995, -41.1968384, 39.5201721
3: -25.4694633, 23.0382729, -31.2481956, 26.7394676, -52.2089310, 54.2864609
4: -18.8570747, 23.4657974, -22.9685173, 27.5716896, -46.4287605, 46.4343147

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8058095, upper bound: 17.8691097
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8025142, upper bound: 17.8668116
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.0587130, 6.7997308, -7.5504246, 6.5225520, -14.5812645, 14.3501549
1: -31.6290512, 26.4341450, -29.5527420, 25.9561958, -57.5852470, 55.9868813
2: -15.4185276, 23.8419552, -14.8029661, 22.4550400, -37.8735657, 38.6449203
3: -27.6192970, 23.9329758, -25.5310497, 23.5824089, -51.2016945, 49.4640236
4: -20.2993298, 24.6619148, -18.9405270, 23.9647255, -44.2640495, 43.6024399

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8506149, upper bound: 17.7862207
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8624026, upper bound: 17.8065475
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.0587130, 6.7997308, -7.3017740, 6.2968907, -14.3556032, 14.1015053
1: -31.6290512, 26.4341450, -28.5910301, 24.9867153, -56.6157646, 55.0251770
2: -15.4185276, 23.8419552, -14.0712976, 21.6516991, -37.0702286, 37.9132538
3: -27.6192970, 23.9329758, -24.7823887, 22.6528416, -50.2721405, 48.7153625
4: -20.2993298, 24.6619148, -18.3272305, 23.0476456, -43.3469734, 42.9891434

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8506150, upper bound: 17.8212540
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8624030, upper bound: 17.8211958
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.4920979, 7.1477027, -7.4961123, 6.4440789, -14.9361753, 14.6438150
1: -33.3568344, 27.7709141, -29.2796936, 25.5022850, -58.8591194, 57.0506058
2: -16.2290630, 25.0428391, -14.6143579, 22.3086681, -38.5377312, 39.6571960
3: -29.1290913, 25.1316872, -25.3306198, 23.1871357, -52.3162193, 50.4623070
4: -21.4046745, 25.8735676, -18.7732849, 23.6385880, -45.0432625, 44.6468506

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8499386, upper bound: 17.7645826
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7786589
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.4920979, 7.1477027, -7.4808340, 6.4002581, -14.8923550, 14.6285362
1: -33.3568344, 27.7709141, -29.2454891, 25.2618828, -58.6187134, 57.0164032
2: -16.2290630, 25.0428391, -14.4270630, 22.1517220, -38.3807831, 39.4699020
3: -29.1290913, 25.1316872, -25.3998356, 22.8915863, -52.0206757, 50.5315247
4: -21.4046745, 25.8735676, -18.7816200, 23.3616943, -44.7663689, 44.6551895

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8499387, upper bound: 17.7919958
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7919369
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.1661940, 6.9140811, -7.3733196, 6.3460736, -14.5122643, 14.2874002
1: -31.9513187, 27.0860863, -28.9331188, 25.0186977, -56.9700165, 56.0191956
2: -15.7361317, 24.1032448, -14.1467009, 21.8121777, -37.5483055, 38.2499466
3: -27.7711315, 24.6224098, -25.1650372, 22.6827507, -50.4538765, 49.7874451
4: -20.5043335, 25.2644825, -18.5705700, 23.0588131, -43.5631485, 43.8350487

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8760823
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8760823
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.1661940, 6.9140811, -8.1638012, 6.9482722, -15.1144629, 15.0778818
1: -31.9513187, 27.0860863, -31.9845276, 27.1597614, -59.1110802, 59.0706062
2: -15.7361317, 24.1032448, -15.6419563, 24.1077423, -39.8438683, 39.7452011
3: -27.7711315, 24.6224098, -27.9125710, 24.5389214, -52.3100510, 52.5349731
4: -20.5043335, 25.2644825, -20.5998497, 25.1355629, -45.6398964, 45.8643303

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8791943
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8791945
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.8005185, 6.6397552, -7.3733196, 6.3460736, -14.1465912, 14.0130730
1: -30.4766083, 26.0748787, -28.9331188, 25.0186977, -55.4953003, 55.0079803
2: -15.0684252, 23.1567383, -14.1467009, 21.8121777, -36.8805962, 37.3034363
3: -26.4369164, 23.7238541, -25.1650372, 22.6827507, -49.1196671, 48.8888931
4: -19.5588493, 24.3224087, -18.5705700, 23.0588131, -42.6176567, 42.8929634

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8636911
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8665161
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.8005185, 6.6397552, -8.1638012, 6.9482722, -14.7487907, 14.8035545
1: -30.4766083, 26.0748787, -31.9845276, 27.1597614, -57.6363678, 58.0593910
2: -15.0684252, 23.1567383, -15.6419563, 24.1077423, -39.1761589, 38.7986946
3: -26.4369164, 23.7238541, -27.9125710, 24.5389214, -50.9758377, 51.6364250
4: -19.5588493, 24.3224087, -20.5998497, 25.1355629, -44.6944084, 44.9222450

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8650960
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8689644
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.3253193, 7.0077128, -9.2867737, 7.7889447, -16.1142635, 16.2944870
1: -32.5765190, 27.3011913, -36.4836769, 30.1828270, -62.7593460, 63.7848663
2: -15.9526958, 24.5148010, -17.6450825, 27.3337231, -43.2864189, 42.1598816
3: -28.3884468, 24.7382965, -31.8809223, 27.2578411, -55.6462860, 56.6192169
4: -20.9383926, 25.4900627, -23.4222240, 28.1255722, -49.0639572, 48.9122849

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206975, upper bound: 17.8611016
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206975, upper bound: 17.8611016
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.3753338, 7.0469851, -9.0874615, 7.6266046, -16.0019379, 16.1344471
1: -32.7698517, 27.4507275, -35.7028503, 29.5892143, -62.3590584, 63.1535759
2: -16.0440159, 24.6538620, -17.2733860, 26.7008476, -42.7448502, 41.9272461
3: -28.5566902, 24.8765202, -31.2111835, 26.7259941, -55.2826843, 56.0877037
4: -21.0623055, 25.6357231, -22.9329090, 27.5346375, -48.5969429, 48.5686302

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8206467, upper bound: 17.8610999
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8203149, upper bound: 17.8522314
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8203149, upper bound: 17.8610999
time: 0.74 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.51 seconds
NS_A1_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.7750767, upper bound: 17.8362228
NS_A1_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8635938
NS_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.7750766, upper bound: 17.8453178
NS_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.7942191, upper bound: 17.8679904
NS_A1_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8117068, upper bound: 17.8573185
NS_A1_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8217249, upper bound: 17.8686260
NS_A1_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8318722, upper bound: 17.8790051
NS_A1_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8332327, upper bound: 17.8792645
NS_A1_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8315257, upper bound: 17.8713028
NS_A1_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8328246, upper bound: 17.8697862
NS_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8054450, upper bound: 17.8629350
NS_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8024683, upper bound: 17.8630315
NS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8058095, upper bound: 17.8691097
NS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8025142, upper bound: 17.8668116
NS_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8506149, upper bound: 17.7862207
NS_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8624026, upper bound: 17.8065475
NS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8506150, upper bound: 17.8212540
NS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8624030, upper bound: 17.8211958
NS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8499386, upper bound: 17.7645826
NS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7786589
NS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8499387, upper bound: 17.7919958
NS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8620742, upper bound: 17.7919369
NS_A2_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8760823
NS_A2_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8760823
NS_A2_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8791943
NS_A2_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8791945
NS_A2_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8636911
NS_A2_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8665161
NS_A2_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8650960
NS_A2_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8689644
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8206975, upper bound: 17.8611016
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8206975, upper bound: 17.8611016
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8203149, upper bound: 17.8522314
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.51
Output dim: 0, lower bound: -17.8203149, upper bound: 17.8610999

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0546312, 6.0894542, -8.1091537, 6.8541374, -13.9087687, 14.1986055
1: -27.6434193, 24.3658524, -31.7455769, 26.8859882, -54.5294075, 56.1114273
2: -13.6369877, 20.8672562, -15.5711403, 23.8847103, -37.5216904, 36.4383965
3: -23.9358082, 22.0362644, -27.6065826, 24.4157505, -48.3515587, 49.6428452
4: -17.7437210, 22.1907616, -20.3740158, 25.0223179, -42.7660332, 42.5647774

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7933986, upper bound: 17.8459313
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.7933986, upper bound: 17.8635938
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.0546312, 6.0894542, -8.3911695, 7.0530047, -14.1076355, 14.4806223
1: -27.6434193, 24.3658524, -32.8637962, 27.5158787, -55.1592979, 57.2296486
2: -13.6369877, 20.8672562, -15.9992733, 24.5585289, -38.1955147, 36.8665314
3: -23.9358082, 22.0362644, -28.6604137, 24.9010277, -48.8368378, 50.6966782
4: -17.7437210, 22.1907616, -21.1387482, 25.5886097, -43.3323212, 43.3295059

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3268852, 6.2685542, -7.0174541, 6.0147562, -13.3416414, 13.2860069
1: -28.7589836, 24.8970490, -27.5345154, 23.5358963, -52.2948647, 52.4315529
2: -14.0606661, 21.4686146, -13.3614292, 20.8268337, -34.8875008, 34.8300438
3: -25.0343590, 22.4018898, -24.0085468, 21.3052273, -46.3395844, 46.4104233
4: -18.5082397, 22.6428070, -17.6468620, 21.7572994, -40.2655411, 40.2896652

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.8511376, 5.9171600, -8.1325941, 6.8638730, -13.7150106, 14.0497541
1: -26.7737122, 23.4296570, -31.9281921, 26.6394501, -53.4131622, 55.3578491
2: -13.2182674, 20.3221054, -15.5028181, 23.9614868, -37.1797562, 35.8249207
3: -23.2540607, 21.2120056, -27.9155197, 24.0500088, -47.3040695, 49.1275253
4: -17.2079639, 21.5711994, -20.5240555, 24.7739277, -41.9818840, 42.0952530

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8315096, upper bound: 17.8692890
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8315096, upper bound: 17.8790051
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3268852, 6.2685542, -7.9135847, 6.6933093, -14.0201950, 14.1821384
1: -28.7589836, 24.8970490, -31.0557060, 26.0012093, -54.7601852, 55.9527512
2: -14.0606661, 21.4686146, -15.0758696, 23.3390865, -37.3997536, 36.5444832
3: -25.0343590, 22.4018898, -27.1578617, 23.4681301, -48.5024834, 49.5597534
4: -18.5082397, 22.6428070, -19.9731731, 24.1524525, -42.6606903, 42.6159782

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8327222, upper bound: 17.8692893
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8327222, upper bound: 17.8792645
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.2240229, 6.2108188, -9.0996466, 7.6320057, -14.8560286, 15.3104649
1: -28.2468643, 24.5834866, -35.7427864, 29.5985088, -57.8453636, 60.3262711
2: -13.9123449, 21.3652401, -17.3090534, 26.7460995, -40.6584435, 38.6742935
3: -24.5462780, 22.2534504, -31.2481956, 26.7394676, -51.2857437, 53.5016441
4: -18.1572723, 22.6486225, -22.9685173, 27.5716896, -45.7289581, 45.6171417

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8216567, upper bound: 17.8615857
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8216567, upper bound: 17.8713027
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.6945238, 6.5606894, -8.8861446, 7.4608731, -15.1553974, 15.4468336
1: -30.2080250, 26.0385609, -34.8808899, 28.9496460, -59.1576691, 60.9194489
2: -14.7547035, 22.5100231, -16.8883266, 26.1305637, -40.8852654, 39.3983498
3: -26.3055592, 23.4368038, -30.4986305, 26.1463165, -52.4518700, 53.9354286
4: -19.4455128, 23.7125206, -22.4220428, 26.9533100, -46.3988228, 46.1345634

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8284713, upper bound: 17.8433337
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8294362, upper bound: 17.8686849
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.1253905, 6.1183491, -8.0203238, 6.7765918, -13.9019814, 14.1386728
1: -27.8663101, 24.1694908, -31.4825649, 26.3068275, -54.1731377, 55.6520538
2: -13.7340384, 21.1555481, -15.2935953, 23.6437283, -37.3777657, 36.4491425
3: -24.1920681, 21.8761463, -27.5261593, 23.7481880, -47.9402504, 49.4023018
4: -17.9205513, 22.3118916, -20.2408791, 24.4576263, -42.3781700, 42.5527573

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8814507, 5.9413695, -8.0808725, 6.8239822, -13.7054319, 14.0222416
1: -26.9047852, 23.5261879, -31.7220154, 26.4892559, -53.3940392, 55.2482033
2: -13.2833033, 20.4562931, -15.4050999, 23.8147984, -37.0981026, 35.8613815
3: -23.3474846, 21.3062592, -27.7346191, 23.9139900, -47.2614746, 49.0408745
4: -17.2995090, 21.6706181, -20.3935719, 24.6297340, -41.9292374, 42.0641861

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.5017118, 6.4201012, -8.9849920, 7.5423079, -15.0440178, 15.4050932
1: -29.3575172, 25.3571835, -35.2894249, 29.2558422, -58.6133575, 60.6465988
2: -14.4495344, 22.2191086, -17.0952320, 26.4229202, -40.8724556, 39.3143387
3: -25.4973545, 22.9508495, -30.8509216, 26.4279156, -51.9252625, 53.8017654
4: -18.8781548, 23.4138699, -22.6789436, 27.2475815, -46.1257324, 46.0928001

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8019079, upper bound: 17.8617323
time: 2.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8019079, upper bound: 17.8668116
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.2625046, 6.2446933, -9.0498743, 7.5930281, -14.8555326, 15.2945671
1: -28.3990211, 24.7246532, -35.5432358, 29.4475422, -57.8465652, 60.2678909
2: -14.0000896, 21.5242424, -17.2150574, 26.6058521, -40.6059418, 38.7392998
3: -24.6549225, 22.3890762, -31.0730972, 26.6044178, -51.2593384, 53.4621735
4: -18.2611523, 22.7766380, -22.8419304, 27.4315701, -45.6927223, 45.6185684

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8019079, upper bound: 17.8617324
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8019079, upper bound: 17.8668116
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.8084750, 6.6009450, -7.4974976, 6.4818997, -14.2903719, 14.0984421
1: -30.6326752, 25.6689644, -29.3416233, 25.8010120, -56.4336739, 55.0105858
2: -14.9466372, 23.1420212, -14.7061119, 22.3102837, -37.2569160, 37.8481255
3: -26.7441921, 23.2518463, -25.3447876, 23.4430199, -50.1872101, 48.5966339
4: -19.6624336, 23.9557686, -18.8047256, 23.8177662, -43.4801941, 42.7604942

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7523984, upper bound: 17.7237260
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7523981, upper bound: 17.7939255
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.8084750, 6.6009450, -7.2451320, 6.2546239, -14.0630980, 13.8460770
1: -30.6326752, 25.6689644, -28.3648796, 24.8251114, -55.4577751, 54.0338440
2: -14.9466372, 23.1420212, -13.9660530, 21.4986439, -36.4452820, 37.1080666
3: -26.7441921, 23.2518463, -24.5821857, 22.5086441, -49.2528381, 47.8340302
4: -19.6624336, 23.9557686, -18.1819229, 22.8960304, -42.5584641, 42.1376915

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8658980, upper bound: 17.8211958
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8658980, upper bound: 17.8211958
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.2409153, 6.9469242, -7.4457626, 6.4038806, -14.6447964, 14.3926849
1: -32.3522606, 27.0019035, -29.0802536, 25.3472672, -57.6995277, 56.0821571
2: -15.7533274, 24.3373566, -14.5198269, 22.1657238, -37.9190483, 38.8571854
3: -28.2463913, 24.4461689, -25.1559544, 23.0472908, -51.2936783, 49.6021233
4: -20.7630730, 25.1626415, -18.6447659, 23.4908829, -44.2539558, 43.8074074

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8404767, upper bound: 17.7447524
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8404767, upper bound: 17.7786589
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.2409153, 6.9469242, -7.4330096, 6.3629179, -14.6038332, 14.3799324
1: -32.3522606, 27.0019035, -29.0559444, 25.1182709, -57.4705315, 56.0578461
2: -15.7533274, 24.3373566, -14.3368835, 22.0150414, -37.7683678, 38.6742401
3: -28.2463913, 24.4461689, -25.2368240, 22.7624817, -51.0088730, 49.6829910
4: -20.7630730, 25.1626415, -18.6621399, 23.2248039, -43.9878769, 43.8247795

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8649103, upper bound: 17.7919369
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.7545409, upper bound: 17.7919369
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.4738464, 7.1824431, -7.3733196, 6.3460736, -14.8199196, 14.5557613
1: -33.2705765, 28.2261162, -28.9331188, 25.0186977, -58.2892761, 57.1592331
2: -16.3402805, 24.9265614, -14.1467009, 21.8121777, -38.1524582, 39.0732574
3: -28.9010773, 25.6526947, -25.1650372, 22.6827507, -51.5838242, 50.8177338
4: -21.2948093, 26.2499218, -18.5705700, 23.0588131, -44.3536148, 44.8204918

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8758788
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8375812, upper bound: 17.8604718
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.2962608, 7.0199008, -7.3733196, 6.3460736, -14.6423340, 14.3932199
1: -32.4650650, 27.5165958, -28.9331188, 25.0186977, -57.4837646, 56.4497070
2: -15.9674244, 24.4455891, -14.1467009, 21.8121777, -37.7796021, 38.5922775
3: -28.2174835, 25.0057373, -25.1650372, 22.6827507, -50.9002342, 50.1707764
4: -20.8367157, 25.6459923, -18.5705700, 23.0588131, -43.8955231, 44.2165527

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376299, upper bound: 17.8758788
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8375812, upper bound: 17.8604718
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.4738464, 7.1824431, -8.1638012, 6.9482722, -15.4221191, 15.3462448
1: -33.2705765, 28.2261162, -31.9845276, 27.1597614, -60.4303360, 60.2106400
2: -16.3402805, 24.9265614, -15.6419563, 24.1077423, -40.4480209, 40.5685196
3: -28.9010773, 25.6526947, -27.9125710, 24.5389214, -53.4399948, 53.5652657
4: -21.2948093, 26.2499218, -20.5998497, 25.1355629, -46.4303703, 46.8497696

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8377058, upper bound: 17.8789400
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376939, upper bound: 17.8738884
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.2962608, 7.0199008, -8.1638012, 6.9482722, -15.2445326, 15.1837025
1: -32.4650650, 27.5165958, -31.9845276, 27.1597614, -59.6248245, 59.5011215
2: -15.9674244, 24.4455891, -15.6419563, 24.1077423, -40.0751648, 40.0875435
3: -28.2174835, 25.0057373, -27.9125710, 24.5389214, -52.7564049, 52.9183083
4: -20.8367157, 25.6459923, -20.5998497, 25.1355629, -45.9722786, 46.2458382

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8377058, upper bound: 17.8789400
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8376939, upper bound: 17.8738885
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.1841545, 6.9724436, -7.3733196, 6.3460736, -14.5302258, 14.3457632
1: -32.0898781, 27.4679146, -28.9331188, 25.0186977, -57.1085739, 56.4010277
2: -15.8117666, 24.1968842, -14.1467009, 21.8121777, -37.6239395, 38.3435860
3: -27.8408775, 24.9756107, -25.1650372, 22.6827507, -50.5236282, 50.1406479
4: -20.5498409, 25.5626469, -18.5705700, 23.0588131, -43.6086464, 44.1332169

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8036981, upper bound: 17.8634821
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8044253, upper bound: 17.8528139
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.8710008, 6.7003369, -7.3733196, 6.3460736, -14.2170734, 14.0736551
1: -30.7534981, 26.3303585, -28.9331188, 25.0186977, -55.7721901, 55.2634697
2: -15.1902523, 23.3437824, -14.1467009, 21.8121777, -37.0024300, 37.4904785
3: -26.6762981, 23.9566460, -25.1650372, 22.6827507, -49.3590469, 49.1216812
4: -19.7365112, 24.5497913, -18.5705700, 23.0588131, -42.7953224, 43.1203575

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8036981, upper bound: 17.8663485
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -17.8035906, upper bound: 17.8579959
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.1841545, 6.9724436, -8.1638012, 6.9482722, -15.1324253, 15.1362448
1: -32.0898781, 27.4679146, -31.9845276, 27.1597614, -59.2496414, 59.4524345
2: -15.8117666, 24.1968842, -15.6419563, 24.1077423, -39.9194984, 39.8388405
3: -27.8408775, 24.9756107, -27.9125710, 24.5389214, -52.3797989, 52.8881836
4: -20.5498409, 25.5626469, -20.5998497, 25.1355629, -45.6853981, 46.1624947

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8034626, upper bound: 17.8650959
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8044918, upper bound: 17.8606597
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.8710008, 6.7003369, -8.1638012, 6.9482722, -14.8192720, 14.8641367
1: -30.7534981, 26.3303585, -31.9845276, 27.1597614, -57.9132538, 58.3148766
2: -15.1902523, 23.3437824, -15.6419563, 24.1077423, -39.2979965, 38.9857407
3: -26.6762981, 23.9566460, -27.9125710, 24.5389214, -51.2152176, 51.8692131
4: -19.7365112, 24.5497913, -20.5998497, 25.1355629, -44.8720741, 45.1496391

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8034626, upper bound: 17.8687854
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -17.8044918, upper bound: 17.8655503
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.4054174, 7.0719872, -9.2867737, 7.7889447, -16.1943626, 16.3587589
1: -32.9033241, 27.6003571, -36.4836769, 30.1828270, -63.0861511, 64.0840302
2: -16.0567455, 24.6359787, -17.6450825, 27.3337231, -43.3904686, 42.2810593
3: -28.6931610, 24.9846516, -31.8809223, 27.2578411, -55.9509964, 56.8655739
4: -21.1659985, 25.6900940, -23.4222240, 28.1255722, -49.2915649, 49.1123199

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.9452462, 6.7295451, -9.2867737, 7.7889447, -15.7341909, 16.0163174
1: -31.0488415, 26.3359890, -36.4836769, 30.1828270, -61.2316666, 62.8196564
2: -15.2198372, 23.4407291, -17.6450825, 27.3337231, -42.5535583, 41.0858116
3: -27.0218983, 23.8580360, -31.8809223, 27.2578411, -54.2797394, 55.7389565
4: -19.9772987, 24.5043411, -23.4222240, 28.1255722, -48.1028633, 47.9265671

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.1696587, 6.8901768, -9.0874615, 7.6266046, -15.7962618, 15.9776363
1: -31.9504986, 26.8285828, -35.7028503, 29.5892143, -61.5397034, 62.5314293
2: -15.6565571, 24.0736485, -17.2733860, 26.7008476, -42.3574066, 41.3470345
3: -27.8372841, 24.3262444, -31.2111835, 26.7259941, -54.5632744, 55.5374298
4: -20.5393295, 25.0510902, -22.9329090, 27.5346375, -48.0739670, 47.9839897

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.94 + 333.82 = 336.76 seconds
