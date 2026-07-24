## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 198.13671952904002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831)
1: (-67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032)
2: (-58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826)
3: (-92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407)
4: (-72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.88 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -198.1763548, upper bound: 198.1763548

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1546311
time: 0.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1546311
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -73.8932495, 135.3371582, -75.5068207, 138.2489624, -212.1422119, 210.8439636
1: -65.9493561, 123.6492538, -67.3784637, 126.3198547, -192.2691650, 191.0277100
2: -57.7101402, 127.8353882, -58.9700966, 130.5795898, -188.2897034, 186.8054810
3: -90.7982178, 126.9703827, -92.7449951, 129.7379456, -220.5361633, 219.7153473
4: -70.8404388, 136.0842285, -72.3862000, 139.0076752, -209.8481140, 208.4704285

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.78 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.58 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -100.2753067, 184.2676697, -71.7902069, 131.6054993, -231.8807983, 256.0578613
1: -90.0813370, 169.2561340, -64.0946198, 120.2725677, -210.3538818, 233.3507385
2: -78.5410080, 174.6634674, -56.0710869, 124.3399887, -202.8809814, 230.7345581
3: -123.8081818, 173.3339539, -88.2992783, 123.4007797, -247.2089539, 261.6332092
4: -96.0769653, 185.9647369, -68.8296967, 132.3437958, -228.4207611, 254.7944336

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.65 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -73.8932495, 135.3371582, -73.8932495, 135.3371582, -209.2304077, 209.2304077
1: -65.9493561, 123.6492538, -65.9493561, 123.6492538, -189.5986023, 189.5986023
2: -57.7101402, 127.8353882, -57.7101402, 127.8353882, -185.5455170, 185.5455170
3: -90.7982178, 126.9703827, -90.7982178, 126.9703827, -217.7686005, 217.7686005
4: -70.8404388, 136.0842285, -70.8404388, 136.0842285, -206.9246674, 206.9246674

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1314898, upper bound: 198.1462241
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1544491
time: 0.89 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -73.8932495, 135.3371582, -100.2753067, 184.2676697, -258.1608276, 235.6124573
1: -65.9493561, 123.6492538, -90.0813370, 169.2561340, -235.2054749, 213.7305756
2: -57.7101402, 127.8353882, -78.5410080, 174.6634674, -232.3736115, 206.3763885
3: -90.7982178, 126.9703827, -123.8081818, 173.3339539, -264.1321716, 250.7785492
4: -70.8404388, 136.0842285, -96.0769653, 185.9647369, -256.8051453, 232.1611938

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1314898, upper bound: 198.1462241
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1544491
time: 0.63 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -100.2753067, 184.2676697, -73.7518311, 135.0833740, -235.3586731, 258.0194702
1: -90.0813370, 169.2561340, -65.8236465, 123.4182816, -213.4995880, 235.0797729
2: -78.5410080, 174.6634674, -57.5982323, 127.5984879, -206.1394958, 232.2617035
3: -123.8081818, 173.3339539, -90.6277237, 126.7278290, -250.5360107, 263.9616699
4: -96.0769653, 185.9647369, -70.7023621, 135.8310852, -231.9080505, 256.6670837

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1381113, upper bound: 198.1341884
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1389268, upper bound: 198.1389268
time: 0.80 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -100.2753067, 184.2676697, -100.2753067, 184.2676697, -284.5429688, 284.5429688
1: -90.0813370, 169.2561340, -90.0813370, 169.2561340, -259.3374329, 259.3374634
2: -78.5410080, 174.6634674, -78.5410080, 174.6634674, -253.2044678, 253.2044678
3: -123.8081818, 173.3339539, -123.8081818, 173.3339539, -297.1421204, 297.1421204
4: -96.0769653, 185.9647369, -96.0769653, 185.9647369, -282.0416565, 282.0416565

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1381113, upper bound: 198.1341884
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1381113, upper bound: 198.1389268
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.60 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1314898, upper bound: 198.1462241
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1544491
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1314898, upper bound: 198.1462241
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1544491
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1381113, upper bound: 198.1341884
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1389268, upper bound: 198.1389268
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1381113, upper bound: 198.1341884
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.60
Output dim: 0, lower bound: -198.1381113, upper bound: 198.1389268

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -67.0086594, 122.8483200, -72.9427261, 133.6090393, -200.6176300, 195.7910461
1: -59.8643456, 112.1512146, -65.1088638, 122.0551453, -181.9194641, 177.2600708
2: -52.3211174, 116.0761490, -56.9658661, 126.2067490, -178.5278473, 173.0419464
3: -82.5262222, 115.0166855, -89.6565552, 125.3176956, -207.8438873, 204.6732483
4: -64.1688080, 123.5948715, -69.9186020, 134.3564453, -198.5252533, 193.5134735

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1749051, upper bound: 198.1747337
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746746, upper bound: 198.1747347
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -75.4596481, 138.1517944, -72.3907089, 132.5750580, -208.0346985, 210.5424652
1: -67.3544464, 126.0165710, -64.6111832, 121.1106567, -188.4651031, 190.6277466
2: -58.9000587, 130.4310303, -56.5242271, 125.2466965, -184.1467590, 186.9552612
3: -92.7729645, 129.5375519, -89.0174866, 124.3432312, -217.1161499, 218.5550385
4: -72.2363968, 138.9253235, -69.3736649, 133.3503113, -205.5867004, 208.2989807

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1722664, upper bound: 198.1730224
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1722614, upper bound: 198.1732132
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -67.0086594, 122.8483200, -99.4451752, 182.7548981, -249.7635040, 222.2934875
1: -59.8643456, 112.1512146, -89.3491287, 167.8705444, -227.7348938, 201.5003357
2: -52.3211174, 116.0761490, -77.8942490, 173.2454376, -225.5665436, 193.9703522
3: -82.5262222, 115.0166855, -122.8225937, 171.8946991, -254.4208984, 237.8392639
4: -64.1688080, 123.5948715, -95.2742233, 184.4654541, -248.6342621, 218.8690948

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -75.4596481, 138.1517944, -98.9066010, 181.7753143, -257.2349548, 237.0583954
1: -67.3544464, 126.0165710, -88.8536987, 166.9465332, -234.3009796, 214.8702698
2: -58.9000587, 130.4310303, -77.4583740, 172.3070526, -231.2071075, 207.8894043
3: -92.7729645, 129.5375519, -122.1375732, 170.9605103, -263.7334290, 251.6750946
4: -72.2363968, 138.9253235, -94.7378311, 183.4536285, -255.6900330, 233.6631470

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1350383, upper bound: 198.1509056
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1347904, upper bound: 198.1510619
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1396325, upper bound: 198.1539258
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -98.0696793, 180.2677612, -73.7514420, 135.0826874, -233.1523590, 254.0191956
1: -88.1109085, 165.6224365, -65.8233109, 123.4176331, -211.5285339, 231.4457397
2: -76.8006668, 170.9264526, -57.5979309, 127.5978394, -204.3984833, 228.5243835
3: -121.1403046, 169.5515747, -90.6272736, 126.7271805, -247.8674927, 260.1787720
4: -93.9258423, 181.9861450, -70.7019882, 135.8303833, -229.7562256, 252.6881409

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1495177, upper bound: 198.1335094
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1495177, upper bound: 198.1335995
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -102.3876572, 188.0603943, -73.6884079, 134.9677582, -237.3554077, 261.7487793
1: -91.9334793, 172.7386475, -65.7673645, 123.3131790, -215.2466583, 238.5059967
2: -80.1782227, 178.2418365, -57.5485382, 127.4906235, -207.6687775, 235.7903748
3: -126.3035736, 176.9449158, -90.5520248, 126.6186218, -252.9221954, 267.4969482
4: -98.1016312, 189.7612305, -70.6409912, 135.7167511, -233.8183594, 260.4022217

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1427605, upper bound: 198.1303598
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1542687, upper bound: 198.1395977
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -98.0696793, 180.2677612, -100.2749023, 184.2668610, -282.3365173, 280.5426636
1: -88.1109085, 165.6224365, -90.0809631, 169.2554321, -257.3663330, 255.7033691
2: -76.8006668, 170.9264526, -78.5406876, 174.6627045, -251.4633636, 249.4671326
3: -121.1403046, 169.5515747, -123.8076782, 173.3332367, -294.4735413, 293.3591919
4: -93.9258423, 181.9861450, -96.0765457, 185.9639893, -279.8898010, 278.0626526

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1339111, upper bound: 198.1339111
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1339111, upper bound: 198.1341884
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -102.3876572, 188.0603943, -100.2172241, 184.1621552, -286.5497742, 288.2775574
1: -91.9334793, 172.7386475, -90.0295792, 169.1598053, -261.0932922, 262.7682190
2: -80.1782227, 178.2418365, -78.4953003, 174.5646820, -254.7428894, 256.7370605
3: -126.3035736, 176.9449158, -123.7378998, 173.2344971, -299.5380249, 300.6827393
4: -98.1016312, 189.7612305, -96.0205612, 185.8595276, -283.9611511, 285.7817078

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1338254, upper bound: 198.1371027
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1365853, upper bound: 198.1374314
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.78 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1749051, upper bound: 198.1747337
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1746746, upper bound: 198.1747347
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1722664, upper bound: 198.1730224
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1722614, upper bound: 198.1732132
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1347904, upper bound: 198.1510619
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1396325, upper bound: 198.1539258
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1495177, upper bound: 198.1335094
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1495177, upper bound: 198.1335995
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1427605, upper bound: 198.1303598
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1542687, upper bound: 198.1395977
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1339111, upper bound: 198.1339111
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1339111, upper bound: 198.1341884
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1338254, upper bound: 198.1371027
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -198.1365853, upper bound: 198.1374314

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -63.4605408, 116.4872437, -61.8877106, 113.9117050, -177.3722534, 178.3749390
1: -56.7229347, 106.3410187, -55.3319855, 104.0058594, -160.7287903, 161.6729736
2: -49.5349274, 110.1108246, -48.2916832, 107.7061005, -157.2410278, 158.4024963
3: -78.2437744, 108.9628906, -76.3905716, 106.4430923, -184.6868591, 185.3534546
4: -60.7400894, 117.2252350, -59.2517738, 114.5798874, -175.3199768, 176.4770050

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708747
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -66.6813049, 122.2704620, -66.9417114, 122.9750900, -189.6563416, 189.2121735
1: -59.5754166, 111.6266785, -59.8079224, 112.3947067, -171.9700928, 171.4345856
2: -52.0650558, 115.5392303, -52.2684937, 116.3118362, -168.3768921, 167.8077240
3: -82.1363907, 114.4618912, -82.5182724, 115.1085281, -197.2448730, 196.9801636
4: -63.8509712, 123.0209198, -64.0848083, 123.7950439, -187.6460114, 187.1056824

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746746, upper bound: 198.1747345
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746746, upper bound: 198.1747345
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -75.4596481, 138.1517944, -59.2406883, 109.1373367, -184.5969849, 197.3924866
1: -67.3544464, 126.0165710, -52.9692726, 99.7275467, -167.0819855, 178.9858398
2: -58.9000587, 130.4310303, -46.2444611, 103.3005905, -162.2006531, 176.6754913
3: -92.7729645, 129.5375519, -73.2680130, 101.8396301, -194.6125946, 202.8055573
4: -72.2363968, 138.9253235, -56.6386681, 109.9412689, -182.1776733, 195.5639801

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691626, upper bound: 198.1723461
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -72.6758728, 133.1898804, -65.7338638, 121.4384308, -194.1143036, 198.9237366
1: -64.8814392, 121.5049515, -58.7419395, 110.8275299, -175.7089386, 180.2468719
2: -56.7214241, 125.7934113, -51.4075813, 114.7246170, -171.4460449, 177.2009888
3: -89.4516678, 124.8022461, -81.1760712, 113.2886963, -202.7403564, 205.9782715
4: -69.5580521, 133.9661255, -63.1049385, 121.9081573, -191.4661713, 197.0710602

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725145, upper bound: 198.1726414
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1725152
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -75.4592667, 138.1510620, -96.7323456, 177.8336029, -253.2928619, 234.8834076
1: -67.3540878, 126.0159378, -86.9098663, 163.3676605, -230.7217407, 212.9258118
2: -58.8997459, 130.4303741, -75.7419739, 168.6255646, -227.5252991, 206.1723328
3: -92.7724915, 129.5368805, -119.5076904, 167.2345886, -260.0069885, 249.0445557
4: -72.2360077, 138.9246368, -92.6160812, 179.5352020, -251.7712097, 231.5407104

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1346626, upper bound: 198.1497232
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -75.3934784, 138.0308380, -101.1052856, 185.7148743, -261.1083374, 239.1361237
1: -67.2958450, 125.9063187, -90.7817688, 170.5618744, -237.8577271, 216.6880798
2: -58.8482513, 130.3181610, -79.1638031, 176.0216980, -234.8699493, 209.4819336
3: -92.6941071, 129.4231110, -124.7375793, 174.7095032, -267.4035950, 254.1606903
4: -72.1723022, 138.8057556, -96.8456497, 187.3998108, -259.5720825, 235.6513977

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1296802, upper bound: 198.1407993
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1395958, upper bound: 198.1539047
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -94.4023972, 173.7561188, -62.8087463, 115.5244598, -209.9268494, 236.5648651
1: -84.8457794, 159.6945190, -56.1323051, 105.6837387, -190.5295105, 215.8268127
2: -73.9201431, 164.8325653, -49.0315132, 109.3665009, -183.2866516, 213.8640747
3: -116.7058182, 163.3706818, -77.5495377, 108.0766373, -224.7824249, 240.9201965
4: -90.3855057, 175.4475098, -60.0993385, 116.3909912, -206.7764893, 235.5468445

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1461642, upper bound: 198.1266405
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1461642, upper bound: 198.1335094
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -95.4085693, 175.4463043, -69.0272598, 127.2564621, -222.6650238, 244.4735718
1: -85.7120819, 161.2289581, -61.6676826, 116.4580917, -202.1701660, 222.8966370
2: -74.6988220, 166.4027100, -53.9419022, 120.2618713, -194.9606934, 220.3446045
3: -117.8740540, 165.0073395, -84.8877411, 118.9913406, -236.8653870, 249.8950500
4: -91.3440552, 177.1457520, -66.2262497, 127.7739258, -219.1179810, 243.3719940

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1266457
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1335995
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -97.1307068, 178.6901398, -63.5413818, 116.9445267, -214.0751953, 242.2314911
1: -87.2565231, 164.2135010, -56.7390137, 106.9552155, -194.2117310, 220.9525146
2: -76.0521469, 169.5000153, -49.5965080, 110.6812668, -186.7334137, 219.0964966
3: -119.9950180, 168.0681915, -78.4567719, 109.5286789, -229.5236969, 246.5249634
4: -93.0211029, 180.4132385, -60.8163567, 117.7615204, -210.7826233, 241.2295990

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -101.5043716, 186.4017029, -71.1533585, 130.1996918, -231.7040710, 257.5549927
1: -91.1449738, 171.2402954, -63.4915085, 119.0209122, -210.1658936, 234.7317810
2: -79.4846497, 176.6761169, -55.5408058, 123.0729446, -202.5575562, 232.2169189
3: -125.2228622, 175.4138794, -87.4835281, 122.2078247, -247.4306946, 262.8973389
4: -97.2540665, 188.1063080, -68.1646194, 131.0533142, -228.3073425, 256.2709351

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1506497, upper bound: 198.1345623
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1530995, upper bound: 198.1381655
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -98.7006760, 181.5274963, -89.6713638, 165.3533173, -264.0539551, 271.1988525
1: -88.6474762, 166.7951202, -80.6267548, 152.0242004, -240.6716461, 247.4218750
2: -77.2799835, 172.1305847, -70.1974945, 156.9648438, -234.2447815, 242.3280792
3: -121.8443527, 170.7478943, -110.9649887, 155.3642120, -277.2085571, 281.7128601
4: -94.5367584, 183.2000275, -85.8391571, 166.9855957, -261.5223389, 269.0391846

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1336330, upper bound: 198.1336330
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1336330, upper bound: 198.1371027
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -99.4768066, 182.7908020, -96.9971619, 178.5940552, -278.0708618, 279.7879333
1: -89.3064728, 167.9451294, -87.0999527, 164.2385864, -253.5450287, 255.0450745
2: -77.8751297, 173.3048248, -75.9370575, 169.3654480, -247.2405701, 249.2418518
3: -122.7338104, 171.9838562, -119.6907883, 167.9101562, -290.6439819, 291.6745300
4: -95.2730103, 184.4804840, -92.9044647, 180.1673584, -275.4403687, 277.3849182

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371027, upper bound: 198.1338254
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371027, upper bound: 198.1374314
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.86 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708747
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1746746, upper bound: 198.1747345
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1746746, upper bound: 198.1747345
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1691626, upper bound: 198.1723461
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1725145, upper bound: 198.1726414
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1725152
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1296802, upper bound: 198.1407993
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1395958, upper bound: 198.1539047
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1461642, upper bound: 198.1266405
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1461642, upper bound: 198.1335094
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1266457
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1335995
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1506497, upper bound: 198.1345623
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1530995, upper bound: 198.1381655
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1336330, upper bound: 198.1336330
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1336330, upper bound: 198.1371027
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1371027, upper bound: 198.1338254
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.86
Output dim: 0, lower bound: -198.1371027, upper bound: 198.1374314

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -53.6915092, 99.2066498, -61.8877106, 113.9117050, -167.6031952, 161.0943604
1: -48.0817757, 90.5729752, -55.3319855, 104.0058594, -152.0876312, 145.9049225
2: -41.9237137, 93.9493637, -48.2916832, 107.7061005, -149.6298065, 142.2410431
3: -66.5944214, 92.2770996, -76.3905716, 106.4430923, -173.0374908, 168.6676636
4: -51.2581024, 99.9797745, -59.2517738, 114.5798874, -165.8379822, 159.2315063

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -57.4453239, 106.7720337, -59.7940407, 110.1793823, -167.6246948, 166.5660706
1: -51.4395142, 97.3888168, -53.4720192, 100.6216278, -152.0611420, 150.8608398
2: -44.9511223, 100.9197311, -46.6614456, 104.2284088, -149.1795349, 147.5811462
3: -71.2238617, 99.1164246, -73.9186554, 102.8678894, -174.0917511, 173.0350800
4: -55.1228790, 107.1250610, -57.2265396, 110.8698120, -165.9926758, 164.3515930

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -58.2753372, 107.4127808, -66.9417114, 122.9750900, -181.2503357, 174.3544922
1: -52.1436195, 98.0396347, -59.8079224, 112.3947067, -164.5383148, 157.8475647
2: -45.4792213, 101.5860672, -52.2684937, 116.3118362, -161.7910614, 153.8545532
3: -72.0490189, 100.1900711, -82.5182724, 115.1085281, -187.1575012, 182.7082977
4: -55.7544632, 108.0647736, -64.0848083, 123.7950439, -179.5494995, 172.1495514

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691555, upper bound: 198.1715848
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725433, upper bound: 198.1725173
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -61.2586174, 112.6507950, -66.9417114, 122.9750900, -184.2336426, 179.5924988
1: -54.7840004, 102.8967514, -59.8079224, 112.3947067, -167.1787109, 162.7046661
2: -47.8160133, 106.5958862, -52.2684937, 116.3118362, -164.1278381, 158.8643799
3: -75.6894073, 105.2293701, -82.5182724, 115.1085281, -190.7979126, 187.7475891
4: -58.5623207, 113.4851379, -64.0848083, 123.7950439, -182.3573608, 177.5699158

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691555, upper bound: 198.1715848
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691555, upper bound: 198.1725149
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -64.3693542, 118.4687881, -56.4086914, 104.0473099, -168.4166412, 174.8774719
1: -57.5703278, 107.9205017, -50.4670219, 95.0547485, -152.6250458, 158.3875122
2: -50.2153435, 111.9017715, -44.0341301, 98.5146561, -148.7299957, 155.9358978
3: -79.4942932, 110.6029739, -69.8745270, 96.9836502, -176.4779358, 180.4775085
4: -61.5503616, 119.1198349, -53.9031715, 104.8425903, -166.3929291, 173.0230103

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -69.0504608, 126.7865143, -58.9847527, 108.6837692, -177.7342072, 185.7712708
1: -61.6904068, 115.6578369, -52.7430840, 99.3142548, -161.0046539, 168.4009247
2: -53.8846016, 119.8329544, -46.0456352, 102.8769608, -156.7614899, 165.8785858
3: -85.1467743, 118.6293335, -72.9640732, 101.4023895, -186.5491638, 191.5933990
4: -66.0166321, 127.6115341, -56.3896446, 109.4893036, -175.5059357, 184.0011749

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -61.7291336, 113.7456284, -60.3227654, 111.7313385, -173.4604797, 174.0683441
1: -55.2254105, 103.6257935, -53.9593544, 101.9413223, -157.1667023, 157.5851440
2: -48.1564789, 107.4894867, -47.1751480, 105.6044998, -153.7609711, 154.6646423
3: -76.3651505, 106.0843887, -74.6757202, 104.0254898, -180.3905640, 180.7600861
4: -59.0033875, 114.4169617, -57.8760948, 112.1678619, -171.1712341, 172.2930603

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725145, upper bound: 198.1726414
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725145, upper bound: 198.1726414
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -66.6639023, 122.5143814, -65.4598618, 120.9547577, -187.6186371, 187.9742432
1: -59.5694618, 111.7795715, -58.4997787, 110.3860550, -169.9555206, 170.2793274
2: -52.0117683, 115.8405609, -51.1939163, 114.2734146, -166.2851715, 167.0344543
3: -82.2974777, 114.5564728, -80.8506699, 112.8211975, -195.1186371, 195.4071350
4: -63.7160492, 123.3515930, -62.8377151, 121.4262619, -185.1422577, 186.1893005

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1725128
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725152, upper bound: 198.1725152
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -68.6592636, 126.1406021, -95.8551559, 176.3500366, -245.0093079, 221.9957428
1: -61.3235550, 115.1254730, -86.1141510, 162.0383911, -223.3619080, 201.2396240
2: -53.5928116, 119.2527313, -75.0447388, 167.2823639, -220.8751831, 194.2974701
3: -84.7339020, 118.1500092, -118.4306030, 165.8388977, -250.5727844, 236.5806122
4: -65.6493759, 126.9940491, -91.7837143, 178.0513000, -243.7006683, 218.7777710

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -71.8560562, 131.4458160, -100.2224884, 184.0552216, -255.9112854, 231.6682892
1: -64.1407928, 119.9837036, -89.9931641, 169.0628052, -233.2035980, 209.9768524
2: -56.0768547, 124.1656494, -78.4707184, 174.4548950, -230.5317383, 202.6363373
3: -88.4116516, 123.3158646, -123.6586533, 173.1781158, -261.5897827, 246.9745178
4: -68.7769852, 132.2969513, -95.9979248, 185.7450104, -254.5220032, 228.2948608

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -62.8087463, 115.5244598, -202.9521179, 224.0630951
1: -78.6254196, 148.2930603, -56.1323051, 105.6837387, -184.3091431, 204.4253387
2: -68.4286423, 153.1336823, -49.0315132, 109.3665009, -177.7951355, 202.1651764
3: -108.2343979, 151.4825134, -77.5495377, 108.0766373, -216.3110199, 229.0320282
4: -83.6595383, 162.9042053, -60.0993385, 116.3909912, -200.0505371, 223.0035400

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264365
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1266538
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -62.8087463, 115.5244598, -210.7055664, 238.0753784
1: -85.4842682, 161.2029877, -56.1323051, 105.6837387, -191.1679993, 217.3352814
2: -74.5097961, 166.2485199, -49.0315132, 109.3665009, -183.8762970, 215.2800293
3: -117.4889450, 164.7580414, -77.5495377, 108.0766373, -225.5655670, 242.3075562
4: -91.1446838, 176.8563690, -60.0993385, 116.3909912, -207.5356750, 236.9556885

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1324837
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1335094
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -69.0272598, 127.2564621, -214.6840973, 230.2816162
1: -78.6254196, 148.2930603, -61.6676826, 116.4580917, -195.0834961, 209.9607239
2: -68.4286423, 153.1336823, -53.9419022, 120.2618713, -188.6905212, 207.0755920
3: -108.2343979, 151.4825134, -84.8877411, 118.9913406, -227.2257080, 236.3702087
4: -83.6595383, 162.9042053, -66.2262497, 127.7739258, -211.4334717, 229.1304474

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264980
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1266457
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -69.0272598, 127.2564621, -222.4375458, 244.2939148
1: -85.4842682, 161.2029877, -61.6676826, 116.4580917, -201.9423370, 222.8706665
2: -74.5097961, 166.2485199, -53.9419022, 120.2618713, -194.7716675, 220.1904297
3: -117.4889450, 164.7580414, -84.8877411, 118.9913406, -236.4802704, 249.6457367
4: -91.1446838, 176.8563690, -66.2262497, 127.7739258, -218.9186096, 243.0825958

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1334611
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1335995
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -90.7785034, 167.3008881, -67.5341873, 123.7364044, -214.5149078, 234.8350525
1: -81.5863800, 153.8351746, -60.2878571, 113.1697922, -194.7561646, 214.1230011
2: -71.0512924, 158.7923126, -52.7076874, 117.0582275, -188.1095276, 211.4999847
3: -112.2179489, 157.2608185, -83.1568375, 116.0577850, -228.2757263, 240.4176178
4: -86.9147339, 168.9089203, -64.6648788, 124.6332550, -211.5479889, 233.5737915

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1479475, upper bound: 198.1335194
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1479475, upper bound: 198.1345375
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -96.1198959, 176.8429565, -68.1714478, 124.7955933, -220.9154663, 245.0143890
1: -86.3009262, 162.5985413, -60.8246880, 114.1208649, -200.4217834, 223.4232330
2: -75.2327271, 167.7139130, -53.1960716, 118.0278854, -193.2606201, 220.9099579
3: -118.5960236, 166.2969208, -83.8925934, 117.0945358, -235.6905518, 250.1894989
4: -92.0449600, 178.4425201, -65.2770767, 125.6814651, -217.7263641, 243.7195892

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1505665, upper bound: 198.1369692
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1505665, upper bound: 198.1381454
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -89.6713638, 165.3533173, -262.3912659, 268.2224731
1: -87.1212387, 164.1400452, -80.6267548, 152.0242004, -239.1454315, 244.7667847
2: -75.9528656, 169.3253326, -70.1974945, 156.9648438, -232.9177094, 239.5228271
3: -119.7224197, 167.8793488, -110.9649887, 155.3642120, -275.0866394, 278.8442993
4: -92.9256363, 180.1524506, -85.8391571, 166.9855957, -259.9112244, 265.9915466

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1253693, upper bound: 198.1351559
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1370924
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -91.6683502, 168.9572144, -96.9971619, 178.5940552, -270.2623901, 265.9543762
1: -82.3829422, 155.3353271, -87.0999527, 164.2385864, -246.6215210, 242.4352722
2: -71.7501450, 160.3607635, -75.9370575, 169.3654480, -241.1155853, 236.2977753
3: -113.3082733, 158.7977295, -119.6907883, 167.9101562, -281.2184448, 278.4884644
4: -87.7642670, 170.5721741, -92.9044647, 180.1673584, -267.9316406, 263.4766235

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1332202
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1253693, upper bound: 198.1338038
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -96.9971619, 178.5940552, -275.6319580, 275.5482788
1: -87.1212387, 164.1400452, -87.0999527, 164.2385864, -251.3598175, 251.2399902
2: -75.9528656, 169.3253326, -75.9370575, 169.3654480, -245.3183136, 245.2623596
3: -119.7224197, 167.8793488, -119.6907883, 167.9101562, -287.6325684, 287.5700378
4: -92.9256363, 180.1524506, -92.9044647, 180.1673584, -273.0929871, 273.0569153

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1365690
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1374314
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.67 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691555, upper bound: 198.1715848
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1725433, upper bound: 198.1725173
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691555, upper bound: 198.1715848
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691555, upper bound: 198.1725149
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1719399
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1725145, upper bound: 198.1726414
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1725145, upper bound: 198.1726414
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1691559, upper bound: 198.1725128
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1725152, upper bound: 198.1725152
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264365
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1266538
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1324837
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1335094
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264980
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1266457
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1334611
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1335995
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1479475, upper bound: 198.1335194
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1479475, upper bound: 198.1345375
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1505665, upper bound: 198.1369692
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1505665, upper bound: 198.1381454
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1253693, upper bound: 198.1351559
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1370924
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1332202
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1253693, upper bound: 198.1338038
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1365690
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.67
Output dim: 0, lower bound: -198.1259283, upper bound: 198.1374314

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -53.6915092, 99.2066498, -58.2753372, 107.4127808, -161.1042633, 157.4819641
1: -48.0817757, 90.5729752, -52.1436195, 98.0396347, -146.1213989, 142.7165833
2: -41.9237137, 93.9493637, -45.4792213, 101.5860672, -143.5097809, 139.4285889
3: -66.5944214, 92.2770996, -72.0490189, 100.1900711, -166.7844391, 164.3260803
4: -51.2581024, 99.9797745, -55.7544632, 108.0647736, -159.3228760, 155.7341766

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -53.6915092, 99.2066498, -64.2876129, 118.3201904, -172.0117035, 163.4942627
1: -48.0817757, 90.5729752, -57.4978485, 107.7766190, -155.8583832, 148.0708313
2: -41.9237137, 93.9493637, -50.1526108, 111.7568893, -153.6805878, 144.1019745
3: -66.5944214, 92.2770996, -79.3972549, 110.4531708, -177.0475922, 171.6743469
4: -51.2581024, 99.9797745, -61.4755058, 118.9662781, -170.2243805, 161.4552460

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708748
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1708747
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -57.4453239, 106.7720337, -56.2288666, 103.7761688, -161.2214966, 163.0008698
1: -51.4395142, 97.3888168, -50.3302116, 94.7364273, -146.1759338, 147.7190247
2: -44.9511223, 100.9197311, -43.8893585, 98.1922836, -143.1434021, 144.8090515
3: -71.2238617, 99.1164246, -69.6464996, 96.6941605, -167.9180298, 168.7629242
4: -55.1228790, 107.1250610, -53.7817345, 104.4464035, -159.5692749, 160.9067993

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1725165
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -57.4453239, 106.7720337, -61.3929138, 113.1375351, -170.5828552, 168.1648865
1: -51.4395142, 97.3888168, -54.9320412, 103.0726852, -154.5121613, 152.3208618
2: -44.9511223, 100.9197311, -47.8983727, 106.9216156, -151.8727417, 148.8180847
3: -71.2238617, 99.1164246, -75.9681549, 105.5043488, -176.7282104, 175.0845795
4: -55.1228790, 107.1250610, -58.6827774, 113.8139877, -168.9368591, 165.8078308

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726678, upper bound: 198.1725165
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1726696, upper bound: 198.1725165
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -58.2753372, 107.4127808, -54.9119911, 101.4977875, -159.7730865, 162.3247681
1: -52.1436195, 98.0396347, -49.1457748, 92.7939377, -144.9375458, 147.1854095
2: -45.4792213, 101.5860672, -42.8858261, 96.1874237, -141.6666412, 144.4718933
3: -72.0490189, 100.1900711, -68.1114731, 94.4619141, -166.5109100, 168.3015137
4: -55.7544632, 108.0647736, -52.4241104, 102.3393631, -158.0938263, 160.4888916

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691875, upper bound: 198.1720710
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691875, upper bound: 198.1723486
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -56.2288666, 103.7761688, -61.0815125, 113.3371887, -169.5660248, 164.8576813
1: -50.3302116, 94.7364273, -54.6349907, 103.4768677, -153.8070831, 149.3713989
2: -43.8893585, 98.1922836, -47.7886200, 107.1765747, -151.0659027, 145.9808655
3: -69.6464996, 96.6941605, -75.6288681, 105.4249268, -175.0713959, 172.3230286
4: -53.7817345, 104.4464035, -58.5782013, 113.7962189, -167.5779572, 163.0245667

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725635, upper bound: 198.1722741
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725635, upper bound: 198.1726884
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -61.2586174, 112.6507950, -54.9119911, 101.4977875, -162.7563934, 167.5627899
1: -54.7840004, 102.8967514, -49.1457748, 92.7939377, -147.5779419, 152.0424957
2: -47.8160133, 106.5958862, -42.8858261, 96.1874237, -144.0034332, 149.4817200
3: -75.6894073, 105.2293701, -68.1114731, 94.4619141, -170.1513214, 173.3408051
4: -58.5623207, 113.4851379, -52.4241104, 102.3393631, -160.9016876, 165.9092407

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691442, upper bound: 198.1703342
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691442, upper bound: 198.1715848
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -59.2364693, 109.0455780, -61.0815125, 113.3371887, -172.5736237, 170.1270905
1: -52.9844704, 99.6129913, -54.6349907, 103.4768677, -156.4613190, 154.2479858
2: -46.2382317, 103.2255478, -47.7886200, 107.1765747, -153.4148102, 151.0141449
3: -73.3032913, 101.7626572, -75.6288681, 105.4249268, -178.7281799, 177.3915100
4: -56.6045914, 109.8959961, -58.5782013, 113.7962189, -170.4008179, 168.4741974

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725401, upper bound: 198.1708729
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725401, upper bound: 198.1725149
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -64.3693542, 118.4687881, -51.9997711, 96.3494186, -160.7187805, 170.4685669
1: -57.5703278, 107.9205017, -46.5976181, 87.9907837, -145.5610962, 154.5180511
2: -50.2153435, 111.9017715, -40.6242676, 91.2687836, -141.4841309, 152.5260315
3: -79.4942932, 110.6029739, -64.6758423, 89.4956818, -168.9899750, 175.2788086
4: -61.5503616, 119.1198349, -49.6951599, 97.0718536, -158.6221924, 168.8150024

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1715831
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1723461
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -64.3693542, 118.4687881, -53.5284233, 98.9602127, -163.3295441, 171.9972076
1: -57.5703278, 107.9205017, -47.9192123, 90.4732285, -148.0435486, 155.8396912
2: -50.2153435, 111.9017715, -41.8006592, 93.8075333, -144.0228577, 153.7024231
3: -79.4942932, 110.6029739, -66.5037613, 92.0404434, -171.5347290, 177.1067352
4: -61.5503616, 119.1198349, -51.0810966, 99.8345108, -161.3848724, 170.2009277

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1715831
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1723461
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -69.0504608, 126.7865143, -51.9997711, 96.3494186, -165.3998718, 178.7862854
1: -61.6904068, 115.6578369, -46.5976181, 87.9907837, -149.6811829, 162.2554016
2: -53.8846016, 119.8329544, -40.6242676, 91.2687836, -145.1533508, 160.4571991
3: -85.1467743, 118.6293335, -64.6758423, 89.4956818, -174.6424561, 183.3051758
4: -66.0166321, 127.6115341, -49.6951599, 97.0718536, -163.0884705, 177.3067017

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1690584
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1719399
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -69.0504608, 126.7865143, -54.0201988, 99.8593597, -168.9098206, 180.8067169
1: -61.6904068, 115.6578369, -48.3543892, 91.2893677, -152.9797668, 164.0122070
2: -53.8846016, 119.8329544, -42.1837997, 94.6493607, -148.5339203, 162.0166931
3: -85.1467743, 118.6293335, -67.0879288, 92.8909683, -178.0377197, 185.7172546
4: -66.0166321, 127.6115341, -51.5542145, 100.7255173, -166.7421265, 179.1657410

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1690584
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1719399
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -61.7291336, 113.7456284, -57.4453239, 106.7720337, -168.5011444, 171.1909027
1: -55.2254105, 103.6257935, -51.4395142, 97.3888168, -152.6142273, 155.0653076
2: -48.1564789, 107.4894867, -44.9511223, 100.9197311, -149.0762024, 152.4406128
3: -76.3651505, 106.0843887, -71.2238617, 99.1164246, -175.4815521, 177.3082581
4: -59.0033875, 114.4169617, -55.1228790, 107.1250610, -166.1284485, 169.5398407

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1716979
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1726411
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -61.7291336, 113.7456284, -60.1662788, 111.2861404, -173.0152588, 173.9118500
1: -55.2254105, 103.6257935, -53.8101387, 101.3285599, -156.5539703, 157.4359131
2: -48.1564789, 107.4894867, -46.9775848, 105.1527023, -153.3091736, 154.4670715
3: -76.3651505, 106.0843887, -74.5205231, 103.5346527, -179.8997803, 180.6048737
4: -59.0033875, 114.4169617, -57.5658722, 111.7905807, -170.7939758, 171.9828339

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1716979
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1691464, upper bound: 198.1726411
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -66.6639023, 122.5143814, -54.5018501, 101.5279236, -168.1917877, 177.0162354
1: -59.5694618, 111.7795715, -48.8480530, 92.6037521, -152.1732178, 160.6275940
2: -52.0117683, 115.8405609, -42.6539879, 96.0182190, -148.0299835, 158.4945526
3: -82.2974777, 114.5564728, -67.7975845, 94.1308441, -176.4282990, 182.3540344
4: -63.7160492, 123.3515930, -52.2876740, 101.9018936, -165.6179199, 175.6392670

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1691559
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1725128
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -66.6639023, 122.5143814, -60.1211166, 111.5301514, -178.1940613, 182.6354980
1: -59.5694618, 111.7795715, -53.7795334, 101.8014145, -161.3708801, 165.5590668
2: -52.0117683, 115.8405609, -47.0321655, 105.4784546, -157.4902039, 162.8727264
3: -82.2974777, 114.5564728, -74.5023193, 103.7198563, -186.0173187, 189.0587921
4: -63.7160492, 123.3515930, -57.6379356, 112.0213013, -175.7373047, 180.9895020

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1691559
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1690584, upper bound: 198.1725128
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -60.6556625, 111.5707245, -198.9983826, 221.9100189
1: -78.6254196, 148.2930603, -54.2129211, 102.0959702, -180.7213898, 202.5059814
2: -68.4286423, 153.1336823, -47.3361053, 105.6864548, -174.1150970, 200.4697723
3: -108.2343979, 151.4825134, -74.9860916, 104.3239899, -212.5583649, 226.4685974
4: -83.6595383, 162.9042053, -57.9867744, 112.4983292, -196.1578674, 220.8909760

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -64.8144684, 119.0687866, -206.4964142, 226.0688324
1: -78.6254196, 148.2930603, -57.9266968, 108.9306641, -187.5560608, 206.2197571
2: -68.4286423, 153.1336823, -50.6168175, 112.7266388, -181.1552734, 203.7505035
3: -108.2343979, 151.4825134, -80.0274353, 111.4796524, -219.7140503, 231.5099487
4: -83.6595383, 162.9042053, -62.0486107, 120.0108566, -203.6703949, 224.9528198

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -60.6556625, 111.5707245, -206.7518311, 235.9223022
1: -85.4842682, 161.2029877, -54.2129211, 102.0959702, -187.5802307, 215.4159088
2: -74.5097961, 166.2485199, -47.3361053, 105.6864548, -180.1962585, 213.5846252
3: -117.4889450, 164.7580414, -74.9860916, 104.3239899, -221.8129272, 239.7441254
4: -91.1446838, 176.8563690, -57.9867744, 112.4983292, -203.6429901, 234.8431091

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -64.8144684, 119.0687866, -214.2498779, 240.0811157
1: -85.4842682, 161.2029877, -57.9266968, 108.9306641, -194.4149170, 219.1296844
2: -74.5097961, 166.2485199, -50.6168175, 112.7266388, -187.2364349, 216.8653412
3: -117.4889450, 164.7580414, -80.0274353, 111.4796524, -228.9685974, 244.7854767
4: -91.1446838, 176.8563690, -62.0486107, 120.0108566, -211.1555328, 238.9049835

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -67.1973801, 123.9003677, -211.3280182, 228.4517365
1: -78.6254196, 148.2930603, -60.0388756, 113.3990707, -192.0244904, 208.3319397
2: -68.4286423, 153.1336823, -52.5032768, 117.1275482, -185.5561676, 205.6369476
3: -108.2343979, 151.4825134, -82.7038040, 115.7961426, -224.0305481, 234.1862946
4: -83.6595383, 162.9042053, -64.4340057, 124.4612274, -208.1207581, 227.3382111

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -69.2396469, 127.4618378, -214.8894806, 230.4940033
1: -78.6254196, 148.2930603, -61.8369408, 116.5977936, -195.2232056, 210.1299896
2: -68.4286423, 153.1336823, -54.0907745, 120.4764404, -188.9050598, 207.2244415
3: -108.2343979, 151.4825134, -85.1761932, 119.2094421, -227.4438324, 236.6586761
4: -83.6595383, 162.9042053, -66.4014816, 128.0751495, -211.7346802, 229.3056946

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -67.1973801, 123.9003677, -219.0814819, 242.4640198
1: -85.4842682, 161.2029877, -60.0388756, 113.3990707, -198.8833313, 221.2418671
2: -74.5097961, 166.2485199, -52.5032768, 117.1275482, -191.6373444, 218.7518005
3: -117.4889450, 164.7580414, -82.7038040, 115.7961426, -233.2850952, 247.4618225
4: -91.1446838, 176.8563690, -64.4340057, 124.4612274, -215.6059113, 241.2903748

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -69.2396469, 127.4618378, -222.6429443, 244.5062866
1: -85.4842682, 161.2029877, -61.8369408, 116.5977936, -202.0820618, 223.0399323
2: -74.5097961, 166.2485199, -54.0907745, 120.4764404, -194.9862366, 220.3392944
3: -117.4889450, 164.7580414, -85.1761932, 119.2094421, -236.6983795, 249.9342041
4: -91.1446838, 176.8563690, -66.4014816, 128.0751495, -219.2198334, 243.2578430

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -90.7785034, 167.3008881, -65.5921326, 120.1702423, -210.9487457, 232.8930206
1: -81.5863800, 153.8351746, -58.5575905, 109.9360504, -191.5223999, 212.3927612
2: -71.0512924, 158.7923126, -51.1715088, 113.7428818, -184.7941742, 209.9638214
3: -112.2179489, 157.2608185, -80.8462830, 112.6781845, -224.8961334, 238.1071014
4: -86.9147339, 168.9089203, -62.7574692, 121.1301956, -208.0449066, 231.6663818

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -90.7785034, 167.3008881, -69.8345947, 127.8248596, -218.6033630, 237.1354675
1: -81.5863800, 153.8351746, -62.3375702, 116.9090805, -198.4954529, 216.1727448
2: -71.0512924, 158.7923126, -54.5253029, 120.9295044, -191.9808044, 213.3175964
3: -112.2179489, 157.2608185, -85.9890747, 119.9633102, -232.1812592, 243.2498474
4: -86.9147339, 168.9089203, -66.8883057, 128.7967987, -215.7115326, 235.7972260

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -96.1198959, 176.8429565, -66.4063721, 121.5569153, -217.6767731, 243.2493134
1: -86.3009262, 162.5985413, -59.2512245, 111.1804886, -197.4814148, 221.8497620
2: -75.2327271, 167.7139130, -51.7996140, 115.0128250, -190.2455444, 219.5135193
3: -118.5960236, 166.2969208, -81.7854309, 114.0255508, -232.6215820, 248.0823517
4: -92.0449600, 178.4425201, -63.5448875, 122.4929810, -214.5379028, 241.9873962

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -96.1198959, 176.8429565, -70.6627655, 129.2194519, -225.3393555, 247.5057220
1: -86.3009262, 162.5985413, -63.0364571, 118.1715240, -204.4724426, 225.6349792
2: -75.2327271, 167.7139130, -55.1542664, 122.2174683, -197.4501953, 222.8681793
3: -118.5960236, 166.2969208, -86.9505692, 121.3215485, -239.9175720, 253.2474976
4: -92.0449600, 178.4425201, -67.6706619, 130.1864471, -222.2313690, 246.1131897

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -91.5342484, 168.7243958, -265.7622681, 270.0853577
1: -87.1212387, 164.1400452, -82.2651749, 155.1216583, -242.2428894, 246.4051666
2: -75.9528656, 169.3253326, -71.6478653, 160.1437988, -236.0966644, 240.9732056
3: -119.7224197, 167.8793488, -113.1501312, 158.5738373, -278.2962341, 281.0294800
4: -92.9256363, 180.1524506, -87.6395721, 170.3393250, -263.2649536, 267.7919922

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -97.0379410, 178.5511017, -275.5890503, 275.5890503
1: -87.1212387, 164.1400452, -87.1212387, 164.1400452, -251.2612610, 251.2612610
2: -75.9528656, 169.3253326, -75.9528656, 169.3253326, -245.2781982, 245.2781982
3: -119.7224197, 167.8793488, -119.7224197, 167.8793488, -287.6017456, 287.6017456
4: -92.9256363, 180.1524506, -92.9256363, 180.1524506, -273.0780945, 273.0780945

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.40 + 261.16 = 264.56 seconds
