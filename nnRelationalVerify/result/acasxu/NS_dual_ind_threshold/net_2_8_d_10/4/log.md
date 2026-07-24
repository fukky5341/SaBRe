## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 53.8414279856


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954)
1: (-20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342)
2: (-34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566)
3: (-40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698)
4: (-30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.83 + 1.56 = 4.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -54.3304016, upper bound: 54.3304016

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1108307, upper bound: 54.3245769
time: 0.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
time: 0.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -54.1108307, upper bound: 54.3245769
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -192.8961639, 191.7658081, -208.9272308, 212.0743713, -404.9705200, 400.6929016
1: -19.1434860, 14.6492538, -20.8862324, 15.8918066, -35.0352936, 35.5354843
2: -32.1835785, 35.5566444, -34.6868706, 39.0282860, -71.2118607, 70.2435150
3: -37.9549179, 23.7596760, -40.3503723, 26.2559071, -64.2108078, 64.1100464
4: -28.4081230, 29.6073837, -30.3582592, 32.2099609, -60.6180725, 59.9656448

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
time: 0.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
time: 0.55 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -187.2980042, 184.6170807, -208.9272308, 212.0743713, -399.3723755, 393.5443115
1: -18.1408882, 14.3323412, -20.8862324, 15.8918066, -34.0326920, 35.2185707
2: -31.0379372, 34.1242867, -34.6868706, 39.0282860, -70.0662155, 68.8111496
3: -36.0846443, 22.8922291, -40.3503723, 26.2559071, -62.3405380, 63.2425995
4: -27.3790359, 27.7733078, -30.3582592, 32.2099609, -59.5889969, 58.1315689

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
time: 0.48 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
time: 0.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.84 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 4, lower bound: -54.1075424, upper bound: 54.1075424

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -192.8961639, 191.7658081, -192.8961639, 191.7658081, -384.6619263, 384.6619263
1: -19.1434860, 14.6492538, -19.1434860, 14.6492538, -33.7927361, 33.7927361
2: -32.1835785, 35.5566444, -32.1835785, 35.5566444, -67.7402191, 67.7402191
3: -37.9549179, 23.7596760, -37.9549179, 23.7596760, -61.7145882, 61.7145882
4: -28.4081230, 29.6073837, -28.4081230, 29.6073837, -58.0154991, 58.0154991

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9740404, upper bound: 54.2629352
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
time: 0.50 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -192.8961639, 191.7658081, -187.2980042, 184.6170807, -377.5132446, 379.0637817
1: -19.1434860, 14.6492538, -18.1408882, 14.3323412, -33.4758263, 32.7901344
2: -32.1835785, 35.5566444, -31.0379372, 34.1242867, -66.3078461, 66.5945816
3: -37.9549179, 23.7596760, -36.0846443, 22.8922291, -60.8471375, 59.8443031
4: -28.4081230, 29.6073837, -27.3790359, 27.7733078, -56.1814270, 56.9864159

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9740404, upper bound: 54.2629352
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
time: 0.55 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -187.2980042, 184.6170807, -192.8961639, 191.7658081, -379.0638123, 377.5132446
1: -18.1408882, 14.3323412, -19.1434860, 14.6492538, -32.7901344, 33.4758263
2: -31.0379372, 34.1242867, -32.1835785, 35.5566444, -66.5945816, 66.3078384
3: -36.0846443, 22.8922291, -37.9549179, 23.7596760, -59.8443146, 60.8471451
4: -27.3790359, 27.7733078, -28.4081230, 29.6073837, -56.9864159, 56.1814270

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9708481, upper bound: 54.0821365
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689725
time: 0.54 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -187.2980042, 184.6170807, -187.2980042, 184.6170807, -371.9151001, 371.9151001
1: -18.1408882, 14.3323412, -18.1408882, 14.3323412, -32.4732208, 32.4732208
2: -31.0379372, 34.1242867, -31.0379372, 34.1242867, -65.1622162, 65.1622162
3: -36.0846443, 22.8922291, -36.0846443, 22.8922291, -58.9768715, 58.9768677
4: -27.3790359, 27.7733078, -27.3790359, 27.7733078, -55.1523438, 55.1523438

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9708481, upper bound: 54.0821365
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689725
time: 0.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.94 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9740404, upper bound: 54.2629352
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9740404, upper bound: 54.2629352
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9708481, upper bound: 54.0821365
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689725
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9708481, upper bound: 54.0821365
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689725

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -174.1565247, 169.0274353, -192.8961639, 191.7658081, -365.9223328, 361.9235840
1: -17.0818806, 13.1804171, -19.1434860, 14.6492538, -31.7311287, 32.3239021
2: -29.1386166, 31.4923210, -32.1835785, 35.5566444, -64.6952591, 63.6758728
3: -34.6955719, 20.9326935, -37.9549179, 23.7596760, -58.4552460, 58.8876076
4: -25.8265209, 26.4998627, -28.4081230, 29.6073837, -55.4339027, 54.9079819

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -557.1829224, 666.6224365, -190.9563446, 189.4611816, -746.6441040, 857.5787964
1: -62.2919769, 42.7310600, -18.9402370, 14.4934263, -76.7854004, 61.6712952
2: -92.3169937, 118.5926971, -31.8706360, 35.1446953, -127.4616699, 150.4633331
3: -101.6287537, 81.8438721, -37.6275520, 23.4691067, -125.0978470, 119.4714203
4: -77.8995361, 95.8493652, -28.1429939, 29.3015118, -107.2010498, 123.9923553

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -174.1565247, 169.0274353, -187.2980042, 184.6170807, -358.7736206, 356.3254395
1: -17.0818806, 13.1804171, -18.1408882, 14.3323412, -31.4142151, 31.3213024
2: -29.1386166, 31.4923210, -31.0379372, 34.1242867, -63.2629013, 62.5302429
3: -34.6955719, 20.9326935, -36.0846443, 22.8922291, -57.5877991, 57.0173378
4: -25.8265209, 26.4998627, -27.3790359, 27.7733078, -53.5998306, 53.8788986

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -557.1829224, 666.6224365, -185.4269409, 182.3777924, -739.5607300, 852.0493774
1: -62.2919769, 42.7310600, -17.9416351, 14.1767445, -76.4687195, 60.6726952
2: -92.3169937, 118.5926971, -30.7375336, 33.7153931, -126.0323868, 149.3302307
3: -101.6287537, 81.8438721, -35.7646866, 22.6090527, -124.2378006, 117.6085587
4: -77.8995361, 95.8493652, -27.1178131, 27.4730988, -105.3726349, 122.9671783

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -170.7095337, 164.2917480, -192.8961639, 191.7658081, -362.4753113, 357.1879272
1: -16.2869015, 12.9607792, -19.1434860, 14.6492538, -30.9361534, 32.1042595
2: -28.3495960, 30.4172039, -32.1835785, 35.5566444, -63.9062424, 62.6007690
3: -33.1597099, 20.3512650, -37.9549179, 23.7596760, -56.9193878, 58.3061790
4: -25.0244255, 24.9520969, -28.4081230, 29.6073837, -54.6318054, 53.3602142

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -525.2477417, 642.0449219, -190.9563446, 189.4611816, -714.7089233, 833.0012207
1: -59.9453850, 41.0192451, -18.9402370, 14.4934263, -74.4388123, 59.9594803
2: -86.9453049, 113.7862854, -31.8706360, 35.1446953, -122.0899963, 145.6569214
3: -97.5886612, 79.0598373, -37.6275520, 23.4691067, -121.0577698, 116.6873932
4: -76.0363998, 91.2880478, -28.1429939, 29.3015118, -105.3379135, 119.4310303

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -170.7095337, 164.2917480, -187.2980042, 184.6170807, -355.3265991, 351.5897217
1: -16.2869015, 12.9607792, -18.1408882, 14.3323412, -30.6192436, 31.1016560
2: -28.3495960, 30.4172039, -31.0379372, 34.1242867, -62.4738770, 61.4551392
3: -33.1597099, 20.3512650, -36.0846443, 22.8922291, -56.0519409, 56.4359016
4: -25.0244255, 24.9520969, -27.3790359, 27.7733078, -52.7977333, 52.3311310

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689724
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689725
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -525.2477417, 642.0449219, -185.4269409, 182.3777924, -707.6254272, 827.4718628
1: -59.9453850, 41.0192451, -17.9416351, 14.1767445, -74.1221313, 58.9608803
2: -86.9453049, 113.7862854, -30.7375336, 33.7153931, -120.6606979, 144.5238037
3: -97.5886612, 79.0598373, -35.7646866, 22.6090527, -120.1977158, 114.8245239
4: -76.0363998, 91.2880478, -27.1178131, 27.4730988, -103.5094986, 118.4058609

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.06 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1355477, upper bound: 54.1355477
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9720031, upper bound: 54.1325171
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -54.1325171, upper bound: 53.9720031
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689724
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9689725, upper bound: 53.9689725
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.06
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -174.1565247, 169.0274353, -174.1565247, 169.0274353, -343.1839600, 343.1839600
1: -17.0818806, 13.1804171, -17.0818806, 13.1804171, -30.2622967, 30.2622967
2: -29.1386166, 31.4923210, -29.1386166, 31.4923210, -60.6309319, 60.6309357
3: -34.6955719, 20.9326935, -34.6955719, 20.9326935, -55.6282654, 55.6282654
4: -25.8265209, 26.4998627, -25.8265209, 26.4998627, -52.3263779, 52.3263855

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1079626, upper bound: 54.1497008
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1112588, upper bound: 54.2489441
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -174.1565247, 169.0274353, -557.1829224, 666.6224365, -840.7789307, 726.2103271
1: -17.0818806, 13.1804171, -62.2919769, 42.7310600, -59.8129311, 75.4723969
2: -29.1386166, 31.4923210, -92.3169937, 118.5926971, -147.7313080, 123.8092957
3: -34.6955719, 20.9326935, -101.6287537, 81.8438721, -116.5394440, 122.5614319
4: -25.8265209, 26.4998627, -77.8995361, 95.8493652, -121.6758881, 104.3993988

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1079626, upper bound: 54.1497008
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1079626, upper bound: 54.2489441
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -557.1829224, 666.6224365, -174.1565247, 169.0274353, -726.2103271, 840.7789307
1: -62.2919769, 42.7310600, -17.0818806, 13.1804171, -75.4723969, 59.8129311
2: -92.3169937, 118.5926971, -29.1386166, 31.4923210, -123.8093033, 147.7313080
3: -101.6287537, 81.8438721, -34.6955719, 20.9326935, -122.5614395, 116.5394440
4: -77.8995361, 95.8493652, -25.8265209, 26.4998627, -104.3993988, 121.6758881

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9680037, upper bound: 54.0942192
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -557.1829224, 666.6224365, -557.1829224, 666.6224365, -1218.2656250, 1218.2657471
1: -62.2919769, 42.7310600, -62.2919769, 42.7310600, -104.6930923, 104.6930923
2: -92.3169937, 118.5926971, -92.3169937, 118.5926971, -210.1862793, 210.1862640
3: -101.6287537, 81.8438721, -101.6287537, 81.8438721, -183.0143127, 183.0143127
4: -77.8995361, 95.8493652, -77.8995361, 95.8493652, -173.2559967, 173.2559967

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9680037, upper bound: 54.0942192
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -174.1565247, 169.0274353, -170.7095337, 164.2917480, -338.4482727, 339.7369690
1: -17.0818806, 13.1804171, -16.2869015, 12.9607792, -30.0426559, 29.4673176
2: -29.1386166, 31.4923210, -28.3495960, 30.4172039, -59.5558167, 59.8419151
3: -34.6955719, 20.9326935, -33.1597099, 20.3512650, -55.0468369, 54.0924034
4: -25.8265209, 26.4998627, -25.0244255, 24.9520969, -50.7786179, 51.5242844

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9256636, upper bound: 54.1472347
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9256636, upper bound: 54.2452342
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -174.1565247, 169.0274353, -525.2477417, 642.0449219, -816.2013550, 694.2751465
1: -17.0818806, 13.1804171, -59.9453850, 41.0192451, -58.1011162, 73.1258011
2: -29.1386166, 31.4923210, -86.9453049, 113.7862854, -142.9248962, 118.4375992
3: -34.6955719, 20.9326935, -97.5886612, 79.0598373, -113.7554092, 118.5213547
4: -25.8265209, 26.4998627, -76.0363998, 91.2880478, -117.1145706, 102.5362625

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9256636, upper bound: 54.1472347
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9278664, upper bound: 54.2452342
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -557.1829224, 666.6224365, -173.6401672, 168.7347870, -725.9177246, 840.2625732
1: -62.2919769, 42.7310600, -16.7079353, 13.2121496, -75.5041275, 59.4389954
2: -92.3169937, 118.5926971, -28.8396435, 31.2237949, -123.5407867, 147.4323425
3: -101.6287537, 81.8438721, -33.7336082, 20.9008179, -122.5295715, 115.5774841
4: -77.8995361, 95.8493652, -25.5071735, 25.6046753, -103.5042038, 121.3565369

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -550.5598755, 658.4807129, -156.5807495, 150.8992004, -701.4591064, 815.0613403
1: -61.5246582, 42.2264214, -14.7747517, 11.9667768, -73.4914322, 57.0011711
2: -91.1919479, 117.1430740, -25.6528740, 27.7435493, -118.9354782, 142.7959442
3: -100.3718567, 80.8541031, -29.3955345, 18.5966110, -118.9684677, 110.2496338
4: -76.9561310, 94.6455307, -22.5653343, 22.0682392, -99.0243683, 117.2108459

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -170.7095337, 164.2917480, -174.1565247, 169.0274353, -339.7369690, 338.4482727
1: -16.2869015, 12.9607792, -17.0818806, 13.1804171, -29.4673157, 30.0426540
2: -28.3495960, 30.4172039, -29.1386166, 31.4923210, -59.8419189, 59.5558205
3: -33.1597099, 20.3512650, -34.6955719, 20.9326935, -54.0924034, 55.0468369
4: -25.0244255, 24.9520969, -25.8265209, 26.4998627, -51.5242844, 50.7786179

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1054827, upper bound: 53.9954384
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1054826, upper bound: 54.0591312
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -170.7095337, 164.2917480, -557.1829224, 666.6224365, -837.3319702, 721.4746094
1: -16.2869015, 12.9607792, -62.2919769, 42.7310600, -59.0179596, 75.2527542
2: -28.3495960, 30.4172039, -92.3169937, 118.5926971, -146.9422913, 122.7341995
3: -33.1597099, 20.3512650, -101.6287537, 81.8438721, -115.0035858, 121.9800186
4: -25.0244255, 24.9520969, -77.8995361, 95.8493652, -120.8737946, 102.8516312

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1054827, upper bound: 53.9954384
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1054827, upper bound: 54.0591312
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -525.2477417, 642.0449219, -174.1565247, 169.0274353, -694.2751465, 816.2013550
1: -59.9453850, 41.0192451, -17.0818806, 13.1804171, -73.1258011, 58.1011162
2: -86.9453049, 113.7862854, -29.1386166, 31.4923210, -118.4375992, 142.9248962
3: -97.5886612, 79.0598373, -34.6955719, 20.9326935, -118.5213547, 113.7554092
4: -76.0363998, 91.2880478, -25.8265209, 26.4998627, -102.5362625, 117.1145706

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9651267, upper bound: 53.9292652
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -525.2477417, 642.0449219, -557.1829224, 666.6224365, -1187.7327881, 1194.6756592
1: -59.9453850, 41.0192451, -62.2919769, 42.7310600, -102.4664154, 103.0215912
2: -86.9453049, 113.7862854, -92.3169937, 118.5926971, -204.9785767, 205.6391144
3: -97.5886612, 79.0598373, -101.6287537, 81.8438721, -179.1042023, 180.4358826
4: -76.0363998, 91.2880478, -77.8995361, 95.8493652, -171.5589447, 168.9827728

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9651267, upper bound: 53.9292652
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -170.7095337, 164.2917480, -170.7095337, 164.2917480, -335.0012207, 335.0012512
1: -16.2869015, 12.9607792, -16.2869015, 12.9607792, -29.2476807, 29.2476807
2: -28.3495960, 30.4172039, -28.3495960, 30.4172039, -58.7667999, 58.7667999
3: -33.1597099, 20.3512650, -33.1597099, 20.3512650, -53.5109749, 53.5109749
4: -25.0244255, 24.9520969, -25.0244255, 24.9520969, -49.9765244, 49.9765244

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9229426, upper bound: 53.9929705
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9229426, upper bound: 54.0555084
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -170.7095337, 164.2917480, -525.2477417, 642.0449219, -812.7544556, 689.5393677
1: -16.2869015, 12.9607792, -59.9453850, 41.0192451, -57.3061447, 72.9061661
2: -28.3495960, 30.4172039, -86.9453049, 113.7862854, -142.1358643, 117.3625107
3: -33.1597099, 20.3512650, -97.5886612, 79.0598373, -112.2195435, 117.9399261
4: -25.0244255, 24.9520969, -76.0363998, 91.2880478, -116.3124695, 100.9884949

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9229426, upper bound: 53.9929705
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9229426, upper bound: 54.0555084
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -525.2477417, 642.0449219, -173.6401672, 168.7347870, -693.9824829, 815.6850586
1: -59.9453850, 41.0192451, -16.7079353, 13.2121496, -73.1575317, 57.7271805
2: -86.9453049, 113.7862854, -28.8396435, 31.2237949, -118.1690903, 142.6259308
3: -97.5886612, 79.0598373, -33.7336082, 20.9008179, -118.4894714, 112.7934418
4: -76.0363998, 91.2880478, -25.5071735, 25.6046753, -101.6410599, 116.7952194

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8966245, upper bound: 53.9075602
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -519.5031738, 634.7997437, -156.5807495, 150.8992004, -670.4023438, 791.3804321
1: -59.2565575, 40.5712128, -14.7747517, 11.9667768, -71.2233276, 55.3459625
2: -85.9641724, 112.5032272, -25.6528740, 27.7435493, -113.7077179, 138.1560822
3: -96.4493027, 78.1741257, -29.3955345, 18.5966110, -115.0459137, 107.5696564
4: -75.1629028, 90.2136307, -22.5653343, 22.0682392, -97.2311401, 112.7789612

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.24 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1079626, upper bound: 54.1497008
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1112588, upper bound: 54.2489441
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1079626, upper bound: 54.1497008
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1079626, upper bound: 54.2489441
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9680037, upper bound: 54.0942192
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9680037, upper bound: 54.0942192
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9256636, upper bound: 54.1472347
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9256636, upper bound: 54.2452342
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9256636, upper bound: 54.1472347
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9278664, upper bound: 54.2452342
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1054827, upper bound: 53.9954384
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1054826, upper bound: 54.0591312
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1054827, upper bound: 53.9954384
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -54.1054827, upper bound: 54.0591312
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9651267, upper bound: 53.9292652
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9651267, upper bound: 53.9292652
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9229426, upper bound: 53.9929705
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9229426, upper bound: 54.0555084
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9229426, upper bound: 53.9929705
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9229426, upper bound: 54.0555084
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.8966245, upper bound: 53.9075602
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.24
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -156.5984497, 144.8957672, -255.3313141, 243.1923828
1: -9.6406240, 8.0492010, -14.9259148, 11.8151102, -21.4557304, 22.9751167
2: -19.1441422, 17.0954742, -26.3855343, 27.3984566, -46.5425987, 43.4810028
3: -24.7065659, 10.9597378, -31.9382133, 18.0532722, -42.7598381, 42.8979492
4: -17.6779728, 14.3685694, -23.6996536, 23.1595078, -40.8374786, 38.0682182

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.2569490, upper bound: 54.1185497
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1696718, upper bound: 54.1177481
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -174.1565247, 169.0274353, -337.7474976, 336.3910217
1: -16.4769020, 12.7550611, -17.0818806, 13.1804171, -29.6573162, 29.8369408
2: -28.2785683, 30.3025055, -29.1386166, 31.4923210, -59.7708893, 59.4411240
3: -33.8082809, 20.1101761, -34.6955719, 20.9326935, -54.7409744, 54.8057480
4: -25.1368675, 25.5685883, -25.8265209, 26.4998627, -51.6367264, 51.3951035

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1604470, upper bound: 54.3024762
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1604470, upper bound: 54.3213232
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -536.5074463, 638.7656860, -749.2012939, 623.1013184
1: -9.6406240, 8.0492010, -59.8072243, 41.1651344, -50.8057518, 67.8564224
2: -19.1441422, 17.0954742, -88.9606628, 113.7955322, -132.9396667, 106.0561295
3: -24.7065659, 10.9597378, -98.2318954, 78.4297638, -103.1363297, 109.1916351
4: -17.6779728, 14.3685694, -75.1373978, 92.0903854, -109.7683563, 89.5059662

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0640462, upper bound: 54.0418040
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.4877514, upper bound: 53.9381357
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -557.1829224, 666.6224365, -835.3424683, 719.4174194
1: -16.4769020, 12.7550611, -62.2919769, 42.7310600, -59.2079620, 75.0470352
2: -28.2785683, 30.3025055, -92.3169937, 118.5926971, -146.8712616, 122.6194992
3: -33.8082809, 20.1101761, -101.6287537, 81.8438721, -115.6521530, 121.7389297
4: -25.1368675, 25.5685883, -77.8995361, 95.8493652, -120.9862289, 103.4681244

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0427808, upper bound: 54.2265514
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0427808, upper bound: 54.2489443
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -174.1565247, 169.0274353, -712.5570679, 823.4377441
1: -60.7894821, 41.6045952, -17.0818806, 13.1804171, -73.8735275, 58.6864662
2: -90.0990143, 115.5417480, -29.1386166, 31.4923210, -121.5913010, 144.6803589
3: -99.1804199, 79.7575989, -34.6955719, 20.9326935, -120.1131134, 114.4531708
4: -76.0097198, 93.5710678, -25.8265209, 26.4998627, -102.5095825, 119.3113632

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0418040, upper bound: 54.0640456
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0882350, upper bound: 54.0691080
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0884419, upper bound: 54.0949584
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -167.5786438, 160.7812500, -677.8241577, 787.8439941
1: -57.8096848, 39.7561531, -16.3093052, 12.6744680, -70.4841461, 56.0654602
2: -85.3252792, 110.3911057, -28.0490761, 30.0295105, -115.3547897, 138.4401855
3: -93.6854935, 76.2938919, -33.4696465, 19.9358177, -113.6212997, 109.7635345
4: -72.1278610, 88.2550354, -24.9365044, 25.2879238, -97.4157791, 113.1915436

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9381357, upper bound: 53.4877514
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0868435, upper bound: 53.9666840
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0870504, upper bound: 53.9667028
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -557.1829224, 666.6224365, -1204.0727539, 1200.1827393
1: -60.7894821, 41.6045952, -62.2919769, 42.7310600, -103.0318146, 103.5353699
2: -90.0990143, 115.5417480, -92.3169937, 118.5926971, -207.8545074, 206.9151154
3: -99.1804199, 79.7575989, -101.6287537, 81.8438721, -180.3955536, 180.8164825
4: -76.0097198, 93.5710678, -77.8995361, 95.8493652, -171.3208466, 170.6795807

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -550.5598755, 658.4807129, -1171.4222412, 1166.0004883
1: -57.8096848, 39.7561531, -61.5246582, 42.2264214, -99.8793411, 101.0549240
2: -85.3252792, 110.3911057, -91.1919479, 117.1430740, -201.9329529, 200.9997559
3: -93.6854935, 76.2938919, -100.3718567, 80.8541031, -174.3501434, 176.2784576
4: -72.1278610, 88.2550354, -76.9561310, 94.6455307, -166.3947449, 165.1845245

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -155.0050964, 141.6517792, -252.0873413, 241.5990295
1: -9.6406240, 8.0492010, -14.2738285, 11.7051182, -21.3457394, 22.3230286
2: -19.1441422, 17.0954742, -25.8424911, 26.5274239, -45.6715622, 42.9379654
3: -24.7065659, 10.9597378, -30.5685749, 17.6225033, -42.3290710, 41.5283127
4: -17.6779728, 14.3685694, -22.9860020, 21.8073673, -39.4853401, 37.3545647

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0480426, upper bound: 54.1149090
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9905598, upper bound: 54.1142172
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -170.7095337, 164.2917480, -333.0117798, 332.9439697
1: -16.4769020, 12.7550611, -16.2869015, 12.9607792, -29.4376793, 29.0419617
2: -28.2785683, 30.3025055, -28.3495960, 30.4172039, -58.6957703, 58.6520996
3: -33.8082809, 20.1101761, -33.1597099, 20.3512650, -54.1595459, 53.2698860
4: -25.1368675, 25.5685883, -25.0244255, 24.9520969, -50.0889664, 50.5930099

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0062887, upper bound: 54.3001489
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0062887, upper bound: 54.3177628
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -506.8710632, 616.2453003, -726.6808472, 593.4649658
1: -9.6406240, 8.0492010, -57.6270180, 39.5904198, -49.2310371, 65.6762161
2: -19.1441422, 17.0954742, -83.9837265, 109.3838959, -128.5280457, 101.0791931
3: -24.7065659, 10.9597378, -94.4861145, 75.8771133, -100.5836792, 105.4458542
4: -17.6779728, 14.3685694, -73.4430084, 87.8173828, -105.4953537, 87.8115768

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8952344, upper bound: 54.0399075
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.4877514, upper bound: 53.9381357
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -525.2477417, 642.0449219, -810.7649536, 687.4821167
1: -16.4769020, 12.7550611, -59.9453850, 41.0192451, -57.4961472, 72.7004471
2: -28.2785683, 30.3025055, -86.9453049, 113.7862854, -142.0648499, 117.2478104
3: -33.8082809, 20.1101761, -97.5886612, 79.0598373, -112.8681183, 117.6988373
4: -25.1368675, 25.5685883, -76.0363998, 91.2880478, -116.4249039, 101.6049805

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9055546, upper bound: 54.2241364
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9055546, upper bound: 54.2452344
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -173.6401672, 168.7347870, -712.2644043, 822.4949341
1: -60.7894821, 41.6045952, -16.7079353, 13.2121496, -73.8584671, 58.3125305
2: -90.0990143, 115.5417480, -28.8396435, 31.2237949, -121.3227997, 144.3305206
3: -99.1804199, 79.7575989, -33.7336082, 20.9008179, -120.0812225, 113.4912109
4: -76.0097198, 93.5710678, -25.5071735, 25.6046753, -101.6143799, 118.9784470

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -173.6401672, 168.7347870, -685.7776489, 793.9055176
1: -57.8096848, 39.7561531, -16.7079353, 13.2121496, -71.0218277, 56.4640884
2: -85.3252792, 110.3911057, -28.8396435, 31.2237949, -116.5490723, 139.2307434
3: -93.6854935, 76.2938919, -33.7336082, 20.9008179, -114.5863037, 110.0274963
4: -72.1278610, 88.2550354, -25.5071735, 25.6046753, -97.7325287, 113.7622070

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -156.5807495, 150.8992004, -694.4288330, 806.0886230
1: -60.7894821, 41.6045952, -14.7747517, 11.9667768, -72.6029968, 56.3793449
2: -90.0990143, 115.5417480, -25.6528740, 27.7435493, -117.8425598, 141.1946106
3: -99.1804199, 79.7575989, -29.3955345, 18.5966110, -117.7770309, 109.1531372
4: -76.0097198, 93.5710678, -22.5653343, 22.0682392, -98.0779572, 116.1363831

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8969801, upper bound: 53.9646690
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -156.5807495, 150.8992004, -667.9421387, 776.8460693
1: -57.8096848, 39.7561531, -14.7747517, 11.9667768, -69.7764511, 54.5309029
2: -85.3252792, 110.3911057, -25.6528740, 27.7435493, -113.0688248, 136.0439758
3: -93.6854935, 76.2938919, -29.3955345, 18.5966110, -112.2821045, 105.6894226
4: -72.1278610, 88.2550354, -22.5653343, 22.0682392, -94.1960983, 110.8203583

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -113.9915314, 91.5321426, -156.5984497, 144.8957672, -258.8872375, 248.1305847
1: -9.7806511, 8.3042717, -14.9259148, 11.8151102, -21.5957603, 23.2301865
2: -19.4328098, 17.4700413, -26.3855343, 27.3984566, -46.8312683, 43.8555717
3: -24.4169292, 11.4310131, -31.9382133, 18.0532722, -42.4701996, 43.3692245
4: -17.7957516, 14.5653934, -23.6996536, 23.1595078, -40.9552574, 38.2650452

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.2530779, upper bound: 53.9636331
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1666663, upper bound: 53.9628053
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -164.5227966, 156.2698212, -174.1565247, 169.0274353, -333.5502319, 330.4263306
1: -15.5817814, 12.4590216, -17.0818806, 13.1804171, -28.7621956, 29.5408974
2: -27.3877792, 29.0134716, -29.1386166, 31.4923210, -58.8801003, 58.1520844
3: -32.1807785, 19.3723373, -34.6955719, 20.9326935, -53.1134720, 54.0679092
4: -24.2404804, 23.8577995, -25.8265209, 26.4998627, -50.7403412, 49.6843185

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1570112, upper bound: 54.0947871
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1570112, upper bound: 54.0971482
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -113.9915314, 91.5321426, -536.5074463, 638.7656860, -752.7572021, 628.0396118
1: -9.7806511, 8.3042717, -59.8072243, 41.1651344, -50.9457855, 68.1114883
2: -19.4328098, 17.4700413, -88.9606628, 113.7955322, -133.2283478, 106.4307022
3: -24.4169292, 11.4310131, -98.2318954, 78.4297638, -102.8466949, 109.6629105
4: -17.7957516, 14.5653934, -75.1373978, 92.0903854, -109.8861389, 89.7027740

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0639039, upper bound: 53.9647564
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.4873518, upper bound: 53.9212100
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -164.5227966, 156.2698212, -557.1829224, 666.6224365, -831.1451416, 713.4527588
1: -15.5817814, 12.4590216, -62.2919769, 42.7310600, -58.3128433, 74.7509995
2: -27.3877792, 29.0134716, -92.3169937, 118.5926971, -145.9804688, 121.3304443
3: -32.1807785, 19.3723373, -101.6287537, 81.8438721, -114.0246506, 121.0010757
4: -24.2404804, 23.8577995, -77.8995361, 95.8493652, -120.0898438, 101.7573395

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0396617, upper bound: 54.0398093
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0396617, upper bound: 54.0591314
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -512.5545654, 626.7749634, -174.1565247, 169.0274353, -681.5819092, 800.9313965
1: -58.5557022, 39.9759598, -17.0818806, 13.1804171, -71.7361221, 57.0578308
2: -84.8863983, 110.9954605, -29.1386166, 31.4923210, -116.3787003, 140.1340790
3: -95.2603912, 77.1416779, -34.6955719, 20.9326935, -116.1930847, 111.8372498
4: -74.2610855, 89.2060852, -25.8265209, 26.4998627, -100.7609406, 115.0326080

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0399075, upper bound: 53.8952344
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0868360, upper bound: 53.9095940
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0868360, upper bound: 53.9095940
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -489.0703735, 600.0740356, -167.5786438, 160.7812500, -649.8516235, 767.6526489
1: -55.8550720, 38.2635422, -16.3093052, 12.6744680, -68.5295258, 54.5728455
2: -80.6301193, 106.3817444, -28.0490761, 30.0295105, -110.6596298, 134.4308167
3: -90.0863113, 73.9556427, -33.4696465, 19.9358177, -110.0221176, 107.4252930
4: -70.3997040, 84.4389801, -24.9365044, 25.2879238, -95.6876221, 109.3754883

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0161741, upper bound: 53.7889295
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0859324, upper bound: 53.8859360
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0829850, upper bound: 53.8858268
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -512.5545654, 626.7749634, -557.1829224, 666.6224365, -1174.6799316, 1177.9261475
1: -58.5557022, 39.9759598, -62.2919769, 42.7310600, -100.9074631, 101.9415131
2: -84.8863983, 110.9954605, -92.3169937, 118.5926971, -202.8587646, 202.5861664
3: -95.2603912, 77.1416779, -101.6287537, 81.8438721, -176.6734772, 178.4016724
4: -74.2610855, 89.2060852, -77.8995361, 95.8493652, -169.7408295, 166.5962830

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -489.0703735, 600.0740356, -550.5598755, 658.4807129, -1144.6416016, 1146.0148926
1: -55.8550720, 38.2635422, -61.5246582, 42.2264214, -97.9858246, 99.5978165
2: -80.6301193, 106.3817444, -91.1919479, 117.1430740, -197.3582764, 197.1688080
3: -90.0863113, 73.9556427, -100.3718567, 80.8541031, -170.8437958, 174.0661774
4: -70.3997040, 84.4389801, -76.9561310, 94.6455307, -164.8097839, 161.3951111

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -113.9915314, 91.5321426, -155.0050964, 141.6517792, -255.6432953, 246.5372314
1: -9.7806511, 8.3042717, -14.2738285, 11.7051182, -21.4857693, 22.5780964
2: -19.4328098, 17.4700413, -25.8424911, 26.5274239, -45.9602318, 43.3125305
3: -24.4169292, 11.4310131, -30.5685749, 17.6225033, -42.0394325, 41.9995880
4: -17.7957516, 14.5653934, -22.9860020, 21.8073673, -39.6031189, 37.5513954

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0442462, upper bound: 53.9600008
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9875408, upper bound: 53.9593115
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -164.5227966, 156.2698212, -170.7095337, 164.2917480, -328.8145142, 326.9793701
1: -15.5817814, 12.4590216, -16.2869015, 12.9607792, -28.5425606, 28.7459221
2: -27.3877792, 29.0134716, -28.3495960, 30.4172039, -57.8049774, 57.3630676
3: -32.1807785, 19.3723373, -33.1597099, 20.3512650, -52.5320396, 52.5320473
4: -24.2404804, 23.8577995, -25.0244255, 24.9520969, -49.1925774, 48.8822250

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0027810, upper bound: 54.0923562
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0027810, upper bound: 54.0936396
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -113.9915314, 91.5321426, -506.8710632, 616.2453003, -730.2368164, 598.4031982
1: -9.7806511, 8.3042717, -57.6270180, 39.5904198, -49.3710709, 65.9312744
2: -19.4328098, 17.4700413, -83.9837265, 109.3838959, -128.8167114, 101.4537659
3: -24.4169292, 11.4310131, -94.4861145, 75.8771133, -100.2940445, 105.9171295
4: -17.7957516, 14.5653934, -73.4430084, 87.8173828, -105.6131363, 88.0083923

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8943276, upper bound: 53.9629802
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.7875007, upper bound: 53.9341003
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -164.5227966, 156.2698212, -525.2477417, 642.0449219, -806.5675659, 681.5175781
1: -15.5817814, 12.4590216, -59.9453850, 41.0192451, -56.6010284, 72.4044037
2: -27.3877792, 29.0134716, -86.9453049, 113.7862854, -141.1740723, 115.9587631
3: -32.1807785, 19.3723373, -97.5886612, 79.0598373, -111.2406158, 116.9609833
4: -24.2404804, 23.8577995, -76.0363998, 91.2880478, -115.5285263, 99.8941956

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9024280, upper bound: 54.0373889
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9024280, upper bound: 54.0555086
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -512.5545654, 626.7749634, -173.6401672, 168.7347870, -681.2892456, 800.2384033
1: -58.5557022, 39.9759598, -16.7079353, 13.2121496, -71.7341156, 56.6838913
2: -84.8863983, 110.9954605, -28.8396435, 31.2237949, -116.1101913, 139.8350830
3: -95.2603912, 77.1416779, -33.7336082, 20.9008179, -116.1612015, 110.8752899
4: -74.2610855, 89.2060852, -25.5071735, 25.6046753, -99.8657455, 114.7132568

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -489.0703735, 600.0740356, -173.6401672, 168.7347870, -657.8051758, 773.7142334
1: -55.8550720, 38.2635422, -16.7079353, 13.2121496, -69.0672226, 54.9714737
2: -80.6301193, 106.3817444, -28.8396435, 31.2237949, -111.8539047, 135.2213898
3: -90.0863113, 73.9556427, -33.7336082, 20.9008179, -110.9871140, 107.6892548
4: -70.3997040, 84.4389801, -25.5071735, 25.6046753, -96.0043793, 109.9461441

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080191
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -512.5545654, 626.7749634, -156.5807495, 150.8992004, -663.4537354, 783.3556519
1: -58.5557022, 39.9759598, -14.7747517, 11.9667768, -70.4786453, 54.7507095
2: -84.8863983, 110.9954605, -25.6528740, 27.7435493, -112.6299438, 136.6483154
3: -95.2603912, 77.1416779, -29.3955345, 18.5966110, -113.8570023, 106.5372162
4: -74.2610855, 89.2060852, -22.5653343, 22.0682392, -96.3293228, 111.7714081

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -489.0703735, 600.0740356, -156.5807495, 150.8992004, -639.9696045, 756.6546631
1: -55.8550720, 38.2635422, -14.7747517, 11.9667768, -67.8218384, 53.0382881
2: -80.6301193, 106.3817444, -25.6528740, 27.7435493, -108.3736572, 132.0346222
3: -90.0863113, 73.9556427, -29.3955345, 18.5966110, -108.6829224, 103.3511810
4: -70.3997040, 84.4389801, -22.5653343, 22.0682392, -92.4679413, 107.0042953

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075605
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
time: 0.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.28 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.2569490, upper bound: 54.1185497
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.1696718, upper bound: 54.1177481
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.1604470, upper bound: 54.3024762
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.1604470, upper bound: 54.3213232
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0640462, upper bound: 54.0418040
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.4877514, upper bound: 53.9381357
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0427808, upper bound: 54.2265514
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0427808, upper bound: 54.2489443
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0882350, upper bound: 54.0691080
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0884419, upper bound: 54.0949584
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0868435, upper bound: 53.9666840
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0870504, upper bound: 53.9667028
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9648154, upper bound: 53.9648154
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0480426, upper bound: 54.1149090
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9905598, upper bound: 54.1142172
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0062887, upper bound: 54.3001489
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0062887, upper bound: 54.3177628
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.8952344, upper bound: 54.0399075
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.4877514, upper bound: 53.9381357
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9055546, upper bound: 54.2241364
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9055546, upper bound: 54.2452344
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9292652, upper bound: 53.9651267
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.8969801, upper bound: 53.9646690
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9077066, upper bound: 53.9646690
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.2530779, upper bound: 53.9636331
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.1666663, upper bound: 53.9628053
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.1570112, upper bound: 54.0947871
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.1570112, upper bound: 54.0971482
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0639039, upper bound: 53.9647564
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.4873518, upper bound: 53.9212100
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0396617, upper bound: 54.0398093
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0396617, upper bound: 54.0591314
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0868360, upper bound: 53.9095940
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0868360, upper bound: 53.9095940
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0859324, upper bound: 53.8859360
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0829850, upper bound: 53.8858268
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9646690, upper bound: 53.9077066
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0442462, upper bound: 53.9600008
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9875408, upper bound: 53.9593115
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0027810, upper bound: 54.0923562
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -54.0027810, upper bound: 54.0936396
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.8943276, upper bound: 53.9629802
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.7875007, upper bound: 53.9341003
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9024280, upper bound: 54.0373889
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9024280, upper bound: 54.0555086
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080179
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9291188, upper bound: 53.9080191
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075605
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.28
Output dim: 4, lower bound: -53.9075450, upper bound: 53.9075602

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -136.8007507, 120.4267426, -230.8623199, 223.3946838
1: -9.6406240, 8.0492010, -12.6903915, 10.1874275, -19.8280506, 20.7395916
2: -19.1441422, 17.0954742, -23.2421970, 22.9905243, -42.1346664, 40.3376694
3: -24.7065659, 10.9597378, -28.5655251, 15.0212746, -39.7278404, 39.5252609
4: -17.6779728, 14.3685694, -20.9745159, 19.6808453, -37.3588181, 35.3430786

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1687791, upper bound: 54.1177481
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1687791, upper bound: 54.1177481
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -109.7355728, 85.7652588, -379.3632812, 437.6342773, -547.3698730, 465.1284790
1: -9.5686312, 7.9907990, -41.7775154, 29.4615746, -39.0301971, 49.7683144
2: -19.0356064, 16.9501266, -61.2522049, 77.7443848, -96.7799911, 78.2023163
3: -24.6083298, 10.8575277, -68.8290787, 53.1938095, -77.8021393, 79.6866074
4: -17.5895271, 14.2481632, -52.9445953, 64.9595795, -82.5491028, 67.1927414

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1687791, upper bound: 54.1177481
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1687791, upper bound: 54.1177481
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -110.4355774, 86.5939560, -255.3139496, 272.6700439
1: -16.4769020, 12.7550611, -9.6406240, 8.0492010, -24.5261021, 22.3956852
2: -28.2785683, 30.3025055, -19.1441422, 17.0954742, -45.3740425, 49.4466476
3: -33.8082809, 20.1101761, -24.7065659, 10.9597378, -44.7680206, 44.8167419
4: -25.1368675, 25.5685883, -17.6779728, 14.3685694, -39.5054359, 43.2465591

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1185497, upper bound: 54.2569487
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1177481, upper bound: 54.1696715
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -168.7200470, 162.2344971, -330.9544983, 330.9545288
1: -16.4769020, 12.7550611, -16.4769020, 12.7550611, -29.2319641, 29.2319622
2: -28.2785683, 30.3025055, -28.2785683, 30.3025055, -58.5810738, 58.5810738
3: -33.8082809, 20.1101761, -33.8082809, 20.1101761, -53.9184570, 53.9184570
4: -25.1368675, 25.5685883, -25.1368675, 25.5685883, -50.7054482, 50.7054482

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1185497, upper bound: 54.2615274
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1177481, upper bound: 54.1708200
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -522.6466064, 621.8002930, -732.2358398, 609.2404785
1: -9.6406240, 8.0492010, -58.2761421, 40.0319633, -49.6725807, 66.2695847
2: -19.1441422, 17.0954742, -86.6891327, 110.7072601, -129.8514099, 103.7845993
3: -24.7065659, 10.9597378, -95.7464371, 76.3152390, -101.0218048, 106.7061768
4: -17.6779728, 14.3685694, -73.2318039, 89.7752838, -107.4532547, 87.6003723

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0240815, upper bound: 54.0326494
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0640456, upper bound: 54.0418040
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.7098618, 82.9885559, -495.7687073, 591.6561890, -698.3660278, 578.7572632
1: -9.2736607, 7.7393479, -55.2620239, 38.1425972, -47.4162598, 63.0013733
2: -18.4873371, 16.3867760, -81.9019165, 105.4424896, -123.9298248, 98.2886963
3: -23.9489670, 10.4806767, -90.2545013, 72.7805405, -96.7295074, 100.7351761
4: -17.1019726, 13.7642183, -69.3187714, 84.4262695, -101.5282440, 83.0829849

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.4694298, upper bound: 53.8900407
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.4693253, upper bound: 53.8843969
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -464.5097351, 542.3038330, -711.0238647, 626.7442627
1: -16.4769020, 12.7550611, -51.1472931, 35.6884727, -52.1653748, 63.9023552
2: -28.2785683, 30.3025055, -77.2320099, 97.1224823, -125.4010468, 107.5345154
3: -33.8082809, 20.1101761, -86.2032776, 66.5877762, -100.3960571, 106.3134537
4: -25.1368675, 25.5685883, -65.4719543, 78.9813538, -104.1182175, 91.0405273

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0423091, upper bound: 54.2251708
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0418375, upper bound: 54.1686826
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -549.9062500, 657.8162231, -826.5362549, 712.1406250
1: -16.4769020, 12.7550611, -61.5091248, 42.1540985, -58.6310005, 74.2641678
2: -28.2785683, 30.3025055, -91.1592789, 117.0207214, -145.2992859, 121.4617844
3: -33.8082809, 20.1101761, -100.3911133, 80.7506638, -114.5589447, 120.5012894
4: -25.1368675, 25.5685883, -76.9303284, 94.6757202, -119.8125839, 102.4989166

Time for backsubstitution: 3.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0423091, upper bound: 54.2472166
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0418375, upper bound: 54.1700027
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -152.2210388, 142.3609924, -685.8906250, 802.0512695
1: -60.7894821, 41.6045952, -14.6177483, 11.4519539, -72.1932831, 56.2223396
2: -90.0990143, 115.5417480, -25.6022453, 26.7132683, -116.8122864, 141.1439972
3: -99.1804199, 79.7575989, -30.8630905, 17.6687241, -116.8491287, 110.6206894
4: -76.0097198, 93.5710678, -22.8345528, 22.7824841, -98.7921982, 116.4056091

Time for backsubstitution: 3.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0518804, upper bound: 54.0195484
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -542.2654419, 648.4379272, -432.6683960, 501.4573364, -1043.7226562, 1075.2525635
1: -60.6616478, 41.4967194, -47.0828629, 33.0393333, -93.3083344, 88.5795822
2: -89.8969193, 115.2725296, -70.4366302, 89.0110092, -178.9079285, 184.9916687
3: -98.9718704, 79.5719223, -78.0788422, 61.3011093, -160.2729492, 157.5231323
4: -75.8376312, 93.3748932, -59.8350639, 72.7000198, -148.5376434, 152.6696167

Time for backsubstitution: 3.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0854948, upper bound: 54.0913136
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -147.0313721, 135.9831085, -653.0260010, 767.2966919
1: -57.8096848, 39.7561531, -14.0245352, 11.0391521, -68.8488159, 53.7806892
2: -85.3252792, 110.3911057, -24.7538719, 25.5807228, -110.9059906, 135.1449738
3: -93.6854935, 76.2938919, -29.9387093, 16.8810291, -110.5665207, 106.2326050
4: -72.1278610, 88.2550354, -22.1508522, 21.8166428, -93.9445038, 110.4058838

Time for backsubstitution: 3.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0504836, upper bound: 53.9659447
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0501249, upper bound: 53.9518735
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -515.4137573, 618.3435059, -414.5962219, 474.4456787, -989.8593750, 1029.2380371
1: -57.6440430, 39.6202545, -45.0671616, 31.7011223, -89.2882385, 84.6874161
2: -85.0647507, 110.0412598, -66.6706161, 84.3134537, -169.3782043, 176.3583221
3: -93.4146118, 76.0515900, -73.1899033, 57.4828339, -150.8973999, 149.2414856
4: -71.9052658, 88.0016403, -55.8516884, 70.1550674, -142.0603180, 143.8533325

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9439994, upper bound: 53.9195355
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0840980, upper bound: 53.9665708
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0837392, upper bound: 53.9524996
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -543.5296631, 649.9139404, -1185.9897461, 1185.9898682
1: -60.7894821, 41.6045952, -60.7894821, 41.6045952, -101.8740921, 101.8740921
2: -90.0990143, 115.5417480, -90.0990143, 115.5417480, -204.5833435, 204.5833435
3: -99.1804199, 79.7575989, -99.1804199, 79.7575989, -178.1977081, 178.1977081
4: -76.0097198, 93.5710678, -76.0097198, 93.5710678, -168.7444458, 168.7444458

Time for backsubstitution: 3.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.4779385, upper bound: 53.5905376
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.4830767, upper bound: 53.7523999
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -543.5296631, 649.9139404, -517.0429077, 620.2653198, -1158.3173828, 1161.3398438
1: -60.7894821, 41.6045952, -57.8096848, 39.7561531, -100.1395035, 99.2185364
2: -90.0990143, 115.5417480, -85.3252792, 110.3911057, -199.7779541, 200.0866547
3: -99.1804199, 79.7575989, -93.6854935, 76.2938919, -174.8929749, 173.1295166
4: -76.0097198, 93.5710678, -72.1278610, 88.2550354, -164.1841431, 164.9744415

Time for backsubstitution: 3.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.4779385, upper bound: 53.5905376
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.4830767, upper bound: 53.7523999
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -543.5296631, 649.9139404, -1161.3398438, 1158.3175049
1: -57.8096848, 39.7561531, -60.7894821, 41.6045952, -99.2185364, 100.1395035
2: -85.3252792, 110.3911057, -90.0990143, 115.5417480, -200.0866547, 199.7779694
3: -93.6854935, 76.2938919, -99.1804199, 79.7575989, -173.1295166, 174.8929749
4: -72.1278610, 88.2550354, -76.0097198, 93.5710678, -164.9744263, 164.1841431

Time for backsubstitution: 3.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9317376, upper bound: 53.9192046
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9324455, upper bound: 53.9324455
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -517.0429077, 620.2653198, -517.0429077, 620.2653198, -1133.6673584, 1133.6674805
1: -57.8096848, 39.7561531, -57.8096848, 39.7561531, -97.4839478, 97.4839478
2: -85.3252792, 110.3911057, -85.3252792, 110.3911057, -195.2812653, 195.2812805
3: -93.6854935, 76.2938919, -93.6854935, 76.2938919, -169.8247986, 169.8247986
4: -72.1278610, 88.2550354, -72.1278610, 88.2550354, -160.3828735, 160.3828735

Time for backsubstitution: 3.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9317376, upper bound: 53.9192046
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9317375, upper bound: 53.9324455
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -135.6225739, 117.2068405, -227.6424103, 222.2165222
1: -9.6406240, 8.0492010, -12.0959034, 10.0899487, -19.7305698, 20.1451035
2: -19.1441422, 17.0954742, -22.7229080, 22.0981007, -41.2422371, 39.8183784
3: -24.7065659, 10.9597378, -27.3967133, 14.5691948, -39.2757607, 38.3564415
4: -17.6779728, 14.3685694, -20.3323975, 18.3837948, -36.0617676, 34.7009583

Time for backsubstitution: 3.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9904724, upper bound: 54.1142172
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9904724, upper bound: 54.1142172
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -109.7355728, 85.7652588, -363.9432373, 415.8914185, -525.6270142, 449.7084656
1: -9.5686312, 7.9907990, -39.1345406, 28.2929287, -37.8615608, 47.1253395
2: -19.0356064, 16.9501266, -57.3721886, 73.6321487, -92.6677551, 74.3223114
3: -24.6083298, 10.8575277, -62.5321007, 50.8545265, -75.4628601, 73.3896179
4: -17.5895271, 14.2481632, -48.9783173, 61.0716362, -78.6611481, 63.2264786

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9904724, upper bound: 54.1142172
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9904724, upper bound: 54.1142172
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -113.9915314, 91.5321426, -260.2521667, 276.2259827
1: -16.4769020, 12.7550611, -9.7806511, 8.3042717, -24.7811737, 22.5357132
2: -28.2785683, 30.3025055, -19.4328098, 17.4700413, -45.7486115, 49.7353134
3: -33.8082809, 20.1101761, -24.4169292, 11.4310131, -45.2392921, 44.5271072
4: -25.1368675, 25.5685883, -17.7957516, 14.5653934, -39.7022629, 43.3643303

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9627932, upper bound: 54.2530776
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9620015, upper bound: 54.1666660
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -168.7200470, 162.2344971, -164.5227966, 156.2698212, -324.9898682, 326.7572327
1: -16.4769020, 12.7550611, -15.5817814, 12.4590216, -28.9359226, 28.3368416
2: -28.2785683, 30.3025055, -27.3877792, 29.0134716, -57.2920380, 57.6902847
3: -33.8082809, 20.1101761, -32.1807785, 19.3723373, -53.1806145, 52.2909546
4: -25.1368675, 25.5685883, -24.2404804, 23.8577995, -48.9946671, 49.8090591

Time for backsubstitution: 3.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9627932, upper bound: 54.2569773
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9620015, upper bound: 54.1672891
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -110.4355774, 86.5939560, -493.7594299, 600.3652344, -710.8007202, 580.3532715
1: -9.6406240, 8.0492010, -56.1748848, 38.5191422, -48.1597519, 64.2240829
2: -19.1441422, 17.0954742, -81.8542175, 106.4878693, -125.6320114, 98.9496841
3: -24.7065659, 10.9597378, -92.0686188, 73.8872833, -98.5938416, 103.0283585
4: -17.6779728, 14.3685694, -71.6039810, 85.6518707, -103.3298416, 85.9725418

Time for backsubstitution: 3.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8937156, upper bound: 54.0305729
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.8952344, upper bound: 54.0399075
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.7098618, 82.9885559, -461.1831970, 561.3520508, -668.0618896, 544.1717529
1: -9.2736607, 7.7393479, -52.3225441, 36.0808411, -45.3544998, 60.0618935
2: -18.4873371, 16.3867760, -76.2155533, 99.6410751, -118.1284103, 92.6023254
3: -23.9489670, 10.4806767, -85.4484024, 69.1911545, -93.1401215, 95.9290771
4: -17.1019726, 13.7642183, -66.5823669, 79.3291168, -96.4310913, 80.3465805

Time for backsubstitution: 3.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.7881787, upper bound: 53.9914569
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.7889295, upper bound: 54.0161741
time: 0.60 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.39 + 416.82 = 421.21 seconds
