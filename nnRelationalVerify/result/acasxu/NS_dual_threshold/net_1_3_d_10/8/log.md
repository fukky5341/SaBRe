## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 187.542370087


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746)
1: (-117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561)
2: (-169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212)
3: (-63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962)
4: (-188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.65 + 1.60 = 2.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6535146, upper bound: 187.8813359
time: 0.58 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6590011
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.18 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -187.6535146, upper bound: 187.8813359
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6590011

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -119.7027283, 108.6696014, -149.6440735, 126.7424088, -246.4451294, 258.3136597
1: -93.7178497, 101.6046600, -117.3338928, 118.4335785, -212.1514282, 218.9385376
2: -135.5732574, 113.5433578, -169.7016296, 131.6250763, -267.1983337, 283.2449646
3: -54.4398117, 137.3926697, -63.3496017, 169.2627869, -223.7026062, 200.7422791
4: -150.9911346, 114.0707092, -188.6523895, 133.4867859, -284.4779053, 302.7230835

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6315550, upper bound: 187.6309287
time: 0.49 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6388782
time: 0.54 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -184.0042267, 156.9276886, -149.6440735, 126.7424088, -310.7466431, 306.5717773
1: -144.4059448, 147.8805542, -117.3338928, 118.4335785, -262.8395386, 265.2144470
2: -208.6540070, 163.4298706, -169.7016296, 131.6250763, -340.2790833, 333.1315002
3: -80.1929092, 207.0820923, -63.3496017, 169.2627869, -249.4556885, 270.4317017
4: -232.1647949, 164.1955109, -188.6523895, 133.4867859, -365.6515808, 352.8478699

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6400201, upper bound: 187.6316936
time: 0.55 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6396430
time: 0.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.70 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.70
Output dim: 3, lower bound: -187.6315550, upper bound: 187.6309287
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.70
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6388782
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.70
Output dim: 3, lower bound: -187.6400201, upper bound: 187.6316936
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.70
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6396430

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -119.7027283, 108.6696014, -119.9573212, 108.7468872, -228.4495850, 228.6269226
1: -93.7178497, 101.6046600, -93.8448410, 101.7164536, -195.4342957, 195.4494934
2: -135.5732574, 113.5433578, -135.7941437, 113.6401825, -249.2134399, 249.3374634
3: -54.4398117, 137.3926697, -54.4526558, 137.4407501, -191.8805542, 191.8453217
4: -150.9911346, 114.0707092, -151.2667084, 114.1601410, -265.1512756, 265.3374023

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
time: 0.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -119.7027283, 108.6696014, -182.1602936, 153.6605072, -273.3632202, 290.8298950
1: -93.7178497, 101.6046600, -143.0398865, 144.6755829, -238.3934326, 244.6445465
2: -135.5732574, 113.5433578, -206.5966339, 159.9870605, -295.5603027, 320.1399536
3: -54.4398117, 137.3926697, -78.3434219, 204.8266907, -259.2665100, 215.7360840
4: -150.9911346, 114.0707092, -229.7526093, 160.7959442, -311.7870789, 343.8233032

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6312251
time: 0.54 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309139, upper bound: 187.6385066
time: 0.50 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -184.0042267, 156.9276886, -119.9573212, 108.7468872, -292.7510986, 276.8849792
1: -144.4059448, 147.8805542, -93.8448410, 101.7164536, -246.1224060, 241.7254028
2: -208.6540070, 163.4298706, -135.7941437, 113.6401825, -322.2941589, 299.2239990
3: -80.1929092, 207.0820923, -54.4526558, 137.4407501, -217.6336670, 261.5347595
4: -232.1647949, 164.1955109, -151.2667084, 114.1601410, -346.3249512, 315.4621887

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289280
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
time: 0.55 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -184.0042267, 156.9276886, -182.1602936, 153.6605072, -337.6647339, 339.0879822
1: -144.4059448, 147.8805542, -143.0398865, 144.6755829, -289.0815125, 290.9204407
2: -208.6540070, 163.4298706, -206.5966339, 159.9870605, -368.6410522, 370.0264893
3: -80.1929092, 207.0820923, -78.3434219, 204.8266907, -285.0195923, 285.4255066
4: -232.1647949, 164.1955109, -229.7526093, 160.7959442, -392.9607544, 393.9481201

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6319900, upper bound: 187.6372789
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6392715
time: 0.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.68 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6312251
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6309139, upper bound: 187.6385066
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289280
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6319900, upper bound: 187.6372789
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.68
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6392715

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -62.5451431, 68.7422791, -119.9573212, 108.7468872, -171.2920227, 188.6996002
1: -49.1084938, 64.2921982, -93.8448410, 101.7164536, -150.8249512, 158.1370392
2: -71.3696976, 72.0312042, -135.7941437, 113.6401825, -185.0098572, 207.8253479
3: -33.7011719, 78.0002060, -54.4526558, 137.4407501, -171.1419220, 132.4528503
4: -79.9041290, 71.9150314, -151.2667084, 114.1601410, -194.0642700, 223.1817322

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295517
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -115.5223923, 105.6780319, -119.9573212, 108.7468872, -224.2692871, 225.6353455
1: -90.4238586, 98.7612991, -93.8448410, 101.7164536, -192.1403198, 192.6061401
2: -130.8580933, 110.4317474, -135.7941437, 113.6401825, -244.4982758, 246.2258911
3: -52.9889755, 132.8598328, -54.4526558, 137.4407501, -190.4297180, 187.3124847
4: -145.7360229, 110.8865128, -151.2667084, 114.1601410, -259.8961792, 262.1531982

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312910, upper bound: 187.6301317
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -119.7027283, 108.6696014, -115.9623871, 108.1399078, -227.8426208, 224.6319885
1: -93.7178497, 101.6046600, -91.0345230, 102.2205048, -195.9383545, 192.6391602
2: -135.5732574, 113.5433578, -131.9047089, 112.9005203, -248.4737396, 245.4480286
3: -54.4398117, 137.3926697, -54.4819260, 135.4607697, -189.9005737, 191.8746033
4: -150.9911346, 114.0707092, -147.2206879, 112.4841080, -263.4752502, 261.2913818

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289039, upper bound: 187.6312251
time: 0.49 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289039, upper bound: 187.6279076
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -119.7027283, 108.6696014, -177.8004608, 150.6191711, -270.3218994, 286.4700623
1: -93.7178497, 101.6046600, -139.5906219, 141.7805023, -235.4983368, 241.1952820
2: -135.5732574, 113.5433578, -201.6681366, 156.8783112, -292.4515686, 315.2114258
3: -54.4398117, 137.3926697, -76.8517990, 200.0969543, -254.5367737, 214.2444763
4: -150.9911346, 114.0707092, -224.2646179, 157.5641174, -308.5552368, 338.3353271

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308966, upper bound: 187.6385066
time: 0.51 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308966, upper bound: 187.6301558
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -117.7962265, 111.3588486, -119.9573212, 108.7468872, -226.5430603, 231.3161621
1: -92.3771362, 105.3934631, -93.8448410, 101.7164536, -194.0935822, 199.2383118
2: -133.9635315, 116.2873993, -135.7941437, 113.6401825, -247.6036835, 252.0815430
3: -56.3376808, 137.7161255, -54.4526558, 137.4407501, -193.7784271, 192.1687775
4: -149.6368866, 115.8507843, -151.2667084, 114.1601410, -263.7970276, 267.1174927

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289039
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289272
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -179.9100342, 154.1128998, -119.9573212, 108.7468872, -288.6568909, 274.0702209
1: -141.1715698, 145.1972504, -93.8448410, 101.7164536, -242.8880157, 239.0420837
2: -204.0370789, 160.5492554, -135.7941437, 113.6401825, -317.6772461, 296.3433838
3: -78.8057022, 202.6504364, -54.4526558, 137.4407501, -216.2464447, 257.1030884
4: -227.0235748, 161.2098389, -151.2667084, 114.1601410, -341.1837158, 312.4765015

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6308965
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -117.7962265, 111.3588486, -182.1602936, 153.6605072, -271.4567261, 293.5191345
1: -92.3771362, 105.3934631, -143.0398865, 144.6755829, -237.0526886, 248.4333496
2: -133.9635315, 116.2873993, -206.5966339, 159.9870605, -293.9505920, 322.8840027
3: -56.3376808, 137.7161255, -78.3434219, 204.8266907, -261.1643372, 216.0595245
4: -149.6368866, 115.8507843, -229.7526093, 160.7959442, -310.4328308, 345.6033936

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289213
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289581
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -179.9100342, 154.1128998, -182.1602936, 153.6605072, -333.5705566, 336.2731934
1: -141.1715698, 145.1972504, -143.0398865, 144.6755829, -285.8471069, 288.2371216
2: -204.0370789, 160.5492554, -206.5966339, 159.9870605, -364.0241394, 367.1458740
3: -78.8057022, 202.6504364, -78.3434219, 204.8266907, -283.6323853, 280.9938660
4: -227.0235748, 161.2098389, -229.7526093, 160.7959442, -387.8195190, 390.9624329

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309139
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309728
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.77 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295517
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6312910, upper bound: 187.6301317
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6289039, upper bound: 187.6312251
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6289039, upper bound: 187.6279076
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6308966, upper bound: 187.6385066
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6308966, upper bound: 187.6301558
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289039
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289272
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6308965
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289213
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289581
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309139
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.77
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309728

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -62.5451431, 68.7422791, -93.5899734, 92.8750916, -155.4202271, 162.3322449
1: -49.1084938, 64.2921982, -73.1107025, 87.0040359, -136.1125183, 137.4028931
2: -71.3696976, 72.0312042, -105.9443512, 97.7353897, -169.1050720, 177.9755402
3: -33.7011719, 78.0002060, -47.2811356, 109.5639648, -143.2651367, 125.2813339
4: -79.9041290, 71.9150314, -118.2672424, 97.3865814, -177.2907104, 190.1822510

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295517
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295517
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -62.5451431, 68.7422791, -149.9745941, 137.3406982, -199.8858337, 218.7168732
1: -49.1084938, 64.2921982, -117.3068619, 129.7829132, -178.8914032, 181.5990601
2: -71.3696976, 72.0312042, -169.9014282, 144.0322723, -215.4018860, 241.9326324
3: -33.7011719, 78.0002060, -70.6789017, 171.0912170, -204.7923889, 148.6791077
4: -79.9041290, 71.9150314, -189.4996490, 143.3141022, -223.2182312, 261.4146423

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -115.5223923, 105.6780319, -93.5899734, 92.8750916, -208.3974915, 199.2680054
1: -90.4238586, 98.7612991, -73.1107025, 87.0040359, -177.4278870, 171.8720093
2: -130.8580933, 110.4317474, -105.9443512, 97.7353897, -228.5934753, 216.3760986
3: -52.9889755, 132.8598328, -47.2811356, 109.5639648, -162.5529480, 180.1409607
4: -145.7360229, 110.8865128, -118.2672424, 97.3865814, -243.1226044, 229.1537323

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301317
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301317
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -115.5223923, 105.6780319, -149.9745941, 137.3406982, -252.8630981, 255.6526184
1: -90.4238586, 98.7612991, -117.3068619, 129.7829132, -220.2067719, 216.0681610
2: -130.8580933, 110.4317474, -169.9014282, 144.0322723, -274.8903198, 280.3331909
3: -52.9889755, 132.8598328, -70.6789017, 171.0912170, -224.0802002, 203.5387268
4: -145.7360229, 110.8865128, -189.4996490, 143.3141022, -289.0501099, 300.3861694

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301558
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301558
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -93.5899734, 92.8750916, -115.9623871, 108.1399078, -201.7298889, 208.8374786
1: -73.1107025, 87.0040359, -91.0345230, 102.2205048, -175.3312073, 178.0385284
2: -105.9443512, 97.7353897, -131.9047089, 112.9005203, -218.8448486, 229.6400757
3: -47.2811356, 109.5639648, -54.4819260, 135.4607697, -182.7418976, 164.0458984
4: -118.2672424, 97.3865814, -147.2206879, 112.4841080, -230.7513428, 244.6072693

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6268885
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6268885
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -115.9623871, 108.1399078, -256.6242676, 250.0097809
1: -116.2964935, 126.5755463, -91.0345230, 102.2205048, -218.5169983, 217.6100464
2: -168.2982788, 140.5842438, -131.9047089, 112.9005203, -281.1987915, 272.4889526
3: -68.7259674, 169.2932587, -54.4819260, 135.4607697, -204.1867218, 223.7751770
4: -187.5333405, 139.8242035, -147.2206879, 112.4841080, -300.0174561, 287.0448914

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6268885
time: 0.49 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6279076
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -93.5899734, 92.8750916, -177.8004608, 150.6191711, -244.2091370, 270.6755371
1: -73.1107025, 87.0040359, -139.5906219, 141.7805023, -214.8911896, 226.5946198
2: -105.9443512, 97.7353897, -201.6681366, 156.8783112, -262.8226624, 299.4035339
3: -47.2811356, 109.5639648, -76.8517990, 200.0969543, -247.3780823, 186.4157715
4: -118.2672424, 97.3865814, -224.2646179, 157.5641174, -275.8312683, 321.6511841

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301490
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301490
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -177.8004608, 150.6191711, -299.1035461, 311.8478394
1: -116.2964935, 126.5755463, -139.5906219, 141.7805023, -258.0769348, 266.1661682
2: -168.2982788, 140.5842438, -201.6681366, 156.8783112, -325.1765747, 342.2523804
3: -68.7259674, 169.2932587, -76.8517990, 200.0969543, -268.8228760, 246.1450348
4: -187.5333405, 139.8242035, -224.2646179, 157.5641174, -345.0974121, 364.0888062

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301317
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301558
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -117.7962265, 111.3588486, -93.5899734, 92.8750916, -210.6713104, 204.9488068
1: -92.3771362, 105.3934631, -73.1107025, 87.0040359, -179.3811646, 178.5041656
2: -133.9635315, 116.2873993, -105.9443512, 97.7353897, -231.6989136, 222.2317505
3: -56.3376808, 137.7161255, -47.2811356, 109.5639648, -165.9016418, 184.9972534
4: -149.6368866, 115.8507843, -118.2672424, 97.3865814, -247.0234680, 234.1180267

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289039
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289039
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -117.7962265, 111.3588486, -149.9745941, 137.3406982, -255.1369171, 261.3334351
1: -92.3771362, 105.3934631, -117.3068619, 129.7829132, -222.1600342, 222.7003174
2: -133.9635315, 116.2873993, -169.9014282, 144.0322723, -277.9957581, 286.1888123
3: -56.3376808, 137.7161255, -70.6789017, 171.0912170, -227.4288788, 208.3950195
4: -149.6368866, 115.8507843, -189.4996490, 143.3141022, -292.9509888, 305.3504333

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289272
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289272
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -179.9100342, 154.1128998, -93.5899734, 92.8750916, -272.7850647, 247.7028809
1: -141.1715698, 145.1972504, -73.1107025, 87.0040359, -228.1755981, 218.3079529
2: -204.0370789, 160.5492554, -105.9443512, 97.7353897, -301.7724609, 266.4935913
3: -78.8057022, 202.6504364, -47.2811356, 109.5639648, -188.3696594, 249.9315796
4: -227.0235748, 161.2098389, -118.2672424, 97.3865814, -324.4100952, 279.4769897

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6308965
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6308965
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -179.9100342, 154.1128998, -149.9745941, 137.3406982, -317.2507324, 304.0874939
1: -141.1715698, 145.1972504, -117.3068619, 129.7829132, -270.9544678, 262.5041199
2: -204.0370789, 160.5492554, -169.9014282, 144.0322723, -348.0693359, 330.4506836
3: -78.8057022, 202.6504364, -70.6789017, 171.0912170, -249.8969116, 273.3293457
4: -227.0235748, 161.2098389, -189.4996490, 143.3141022, -370.3376770, 350.7094727

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6309206
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6309206
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -117.7962265, 111.3588486, -149.2839355, 134.6893616, -252.4855194, 260.6427612
1: -92.3771362, 105.3934631, -116.9905624, 127.1221771, -219.4993134, 222.3840332
2: -133.9635315, 116.2873993, -169.2326202, 141.1778259, -275.1413574, 285.5200195
3: -56.3376808, 137.7161255, -69.0598907, 170.1712341, -226.5088806, 206.7760162
4: -149.6368866, 115.8507843, -188.5170746, 140.5437622, -290.1806030, 304.3678589

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289213
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289039
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -117.7962265, 111.3588486, -214.7231140, 185.9732056, -303.7694397, 326.0819397
1: -92.3771362, 105.3934631, -168.6910553, 175.9269562, -268.3040466, 274.0844727
2: -133.9635315, 116.2873993, -243.5518188, 193.9766388, -327.9401855, 359.8392029
3: -56.3376808, 137.7161255, -95.6573029, 241.0182190, -297.3558350, 232.8351593
4: -149.6368866, 115.8507843, -271.0773621, 193.9729767, -343.6098328, 386.9281311

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289581
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289272
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -179.9100342, 154.1128998, -149.2839355, 134.6893616, -314.5993652, 303.3967896
1: -141.1715698, 145.1972504, -116.9905624, 127.1221771, -268.2937317, 262.1878052
2: -204.0370789, 160.5492554, -169.2326202, 141.1778259, -345.2149048, 329.7818604
3: -78.8057022, 202.6504364, -69.0598907, 170.1712341, -248.9769287, 271.7103271
4: -227.0235748, 161.2098389, -188.5170746, 140.5437622, -367.5673218, 349.7268982

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6309139
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6308965
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -179.9100342, 154.1128998, -214.7231140, 185.9732056, -365.8832397, 368.8359680
1: -141.1715698, 145.1972504, -168.6910553, 175.9269562, -317.0984802, 313.8882141
2: -204.0370789, 160.5492554, -243.5518188, 193.9766388, -398.0137329, 404.1010742
3: -78.8057022, 202.6504364, -95.6573029, 241.0182190, -319.8239136, 297.5856018
4: -227.0235748, 161.2098389, -271.0773621, 193.9729767, -420.9965210, 432.2871704

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6309729
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6309206
time: 0.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.82 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295517
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295517
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301317
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301317
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301558
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6301558
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6268885
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6268885
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6268885
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6279076
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301490
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301490
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301317
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6301558
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289039
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289039
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289272
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6289272
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6308965
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6308965
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6309206
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6309206
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289213
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289039
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289581
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6289272
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6309139
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6308965
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6309729
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.82
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6309206

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -93.5899734, 92.8750916, -133.8560791, 148.3690796
1: -32.1188278, 51.3675804, -73.1107025, 87.0040359, -119.1228638, 124.4782867
2: -47.0364113, 57.8066292, -105.9443512, 97.7353897, -144.7717896, 163.7509766
3: -27.2729225, 55.0080185, -47.2811356, 109.5639648, -136.8368835, 102.2891464
4: -53.0424118, 57.1621284, -118.2672424, 97.3865814, -150.4289856, 175.4293213

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8151774, upper bound: 187.3663697
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8144769, upper bound: 187.5068514
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -87.3655396, 92.1814575, -93.5899734, 92.8750916, -180.2406311, 185.7714081
1: -68.5108109, 87.2906647, -73.1107025, 87.0040359, -155.5148315, 160.4013519
2: -99.3974304, 96.7410889, -105.9443512, 97.7353897, -197.1328125, 202.6854248
3: -46.8131294, 105.1580734, -47.2811356, 109.5639648, -156.3770905, 152.4392090
4: -111.3977280, 95.6210403, -118.2672424, 97.3865814, -208.7843018, 213.8882599

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3509328, upper bound: 187.4854176
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8205051, upper bound: 187.5295517
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -149.9745941, 137.3406982, -178.3216705, 204.7537079
1: -32.1188278, 51.3675804, -117.3068619, 129.7829132, -161.9017334, 168.6744385
2: -47.0364113, 57.8066292, -169.9014282, 144.0322723, -191.0686646, 227.7080383
3: -27.2729225, 55.0080185, -70.6789017, 171.0912170, -198.3641357, 125.6869202
4: -53.0424118, 57.1621284, -189.4996490, 143.3141022, -196.3565063, 246.6617737

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -87.3655396, 92.1814575, -149.9745941, 137.3406982, -224.7062378, 242.1560516
1: -68.5108109, 87.2906647, -117.3068619, 129.7829132, -198.2937317, 204.5975037
2: -99.3974304, 96.7410889, -169.9014282, 144.0322723, -243.4296570, 266.6424866
3: -46.8131294, 105.1580734, -70.6789017, 171.0912170, -217.9043427, 175.8369751
4: -111.3977280, 95.6210403, -189.4996490, 143.3141022, -254.7118225, 285.1206970

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -93.5899734, 92.8750916, -182.3230438, 183.4718781
1: -69.8412399, 84.1715393, -73.1107025, 87.0040359, -156.8452454, 157.2822418
2: -101.2739258, 94.6007919, -105.9443512, 97.7353897, -199.0093079, 200.5451355
3: -45.8811989, 105.0105896, -47.2811356, 109.5639648, -155.4451599, 152.2917175
4: -113.0568390, 94.2403870, -118.2672424, 97.3865814, -210.4434204, 212.5076294

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446681, upper bound: 187.6238230
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209595, upper bound: 187.6301112
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -93.5899734, 92.8750916, -237.1418610, 224.8199463
1: -112.9824219, 123.8839874, -73.1107025, 87.0040359, -199.9864502, 196.9946899
2: -163.5588074, 137.6753693, -105.9443512, 97.7353897, -261.2941895, 243.6196747
3: -67.3619995, 164.7313690, -47.2811356, 109.5639648, -176.9259644, 212.0125122
4: -182.2491913, 136.8414307, -118.2672424, 97.3865814, -279.6357422, 255.1085968

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446681, upper bound: 187.6238230
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209595, upper bound: 187.6301112
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -149.9745941, 137.3406982, -226.7886353, 239.8565063
1: -69.8412399, 84.1715393, -117.3068619, 129.7829132, -199.6241455, 201.4783783
2: -101.2739258, 94.6007919, -169.9014282, 144.0322723, -245.3061523, 264.5022278
3: -45.8811989, 105.0105896, -70.6789017, 171.0912170, -216.9724121, 175.6894836
4: -113.0568390, 94.2403870, -189.4996490, 143.3141022, -256.3709412, 283.7400513

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6279076
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6287846
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -149.9745941, 137.3406982, -281.6074829, 281.2045898
1: -112.9824219, 123.8839874, -117.3068619, 129.7829132, -242.7653198, 241.1908569
2: -163.5588074, 137.6753693, -169.9014282, 144.0322723, -307.5910645, 307.5767822
3: -67.3619995, 164.7313690, -70.6789017, 171.0912170, -238.4532166, 235.4102783
4: -182.2491913, 136.8414307, -189.4996490, 143.3141022, -325.5632935, 326.3410645

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6279076
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6287846
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -93.5899734, 92.8750916, -88.6077271, 92.8103714, -186.4003296, 181.4828186
1: -73.1107025, 87.0040359, -69.5185242, 87.8645935, -160.9752960, 156.5225525
2: -105.9443512, 97.7353897, -100.9003143, 97.3328018, -203.2771454, 198.6357117
3: -47.2811356, 109.5639648, -47.0367928, 106.5991135, -153.8802490, 156.6007538
4: -118.2672424, 97.3865814, -113.0341187, 96.2283783, -214.4955750, 210.4206848

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4854176, upper bound: 187.3509328
time: 0.48 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.8205051
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -93.5899734, 92.8750916, -143.9209900, 137.5663147, -230.3369598, 236.7960815
1: -73.1107025, 87.0040359, -112.8730698, 130.6749878, -202.0905457, 199.8771057
2: -105.9443512, 97.7353897, -163.7390137, 143.6699066, -247.6691437, 261.4743958
3: -47.2811356, 109.5639648, -70.0665512, 166.9166107, -214.1977386, 177.7084351
4: -118.2672424, 97.3865814, -182.9464722, 142.4316864, -259.8266907, 280.3330383

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5249229, upper bound: 187.7697195
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.8122145
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -88.6077271, 92.8103714, -241.2947388, 222.6551208
1: -116.2964935, 126.5755463, -69.5185242, 87.8645935, -204.1610870, 196.0940704
2: -168.2982788, 140.5842438, -100.9003143, 97.3328018, -265.6310730, 241.4845581
3: -68.7259674, 169.2932587, -47.0367928, 106.5991135, -175.3250732, 216.3300323
4: -187.5333405, 139.8242035, -113.0341187, 96.2283783, -283.7617188, 252.8583221

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663784, upper bound: 187.6259274
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068601, upper bound: 187.6252269
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -143.9209900, 137.5663147, -285.6214905, 277.9683838
1: -116.2964935, 126.5755463, -112.8730698, 130.6749878, -245.5727844, 239.4486084
2: -168.2982788, 140.5842438, -163.7390137, 143.6699066, -310.4308777, 304.3232422
3: -68.7259674, 169.2932587, -70.0665512, 166.9166107, -235.6425476, 237.6422882
4: -187.5333405, 139.8242035, -182.9464722, 142.4316864, -329.6094971, 322.7706909

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5273276
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5263085, upper bound: 187.6279076
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -93.5899734, 92.8750916, -144.8092651, 131.6590424, -225.2490234, 237.6843567
1: -73.1107025, 87.0040359, -113.4549103, 124.2506790, -197.3613892, 200.4589539
2: -105.9443512, 97.7353897, -164.1941833, 138.0722961, -244.0166168, 261.9295654
3: -47.2811356, 109.5639648, -67.5839310, 165.3264618, -212.6076050, 177.1478882
4: -118.2672424, 97.3865814, -182.9160309, 137.3219604, -255.5891418, 280.3026123

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6238230, upper bound: 187.4446807
time: 0.51 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301112, upper bound: 187.8209721
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -93.5899734, 92.8750916, -210.2807465, 183.0075378, -276.5975037, 303.1558228
1: -73.1107025, 87.0040359, -165.1631775, 173.1105804, -246.2212830, 252.1671906
2: -105.9443512, 97.7353897, -238.5625763, 190.9392853, -296.8836365, 336.2979126
3: -47.2811356, 109.5639648, -94.2064362, 236.2539673, -283.5350952, 202.9987946
4: -118.2672424, 97.3865814, -265.5162659, 190.8201752, -309.0874023, 362.9028320

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6238230, upper bound: 187.4446807
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301112, upper bound: 187.8209721
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -144.8092651, 131.6590424, -280.1434326, 278.8566589
1: -116.2964935, 126.5755463, -113.4549103, 124.2506790, -240.5471802, 240.0304565
2: -168.2982788, 140.5842438, -164.1941833, 138.0722961, -306.3705750, 304.7784424
3: -68.7259674, 169.2932587, -67.5839310, 165.3264618, -234.0523987, 236.8771973
4: -187.5333405, 139.8242035, -182.9160309, 137.3219604, -324.8552246, 322.7402344

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5295517
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5263085, upper bound: 187.6279424
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -210.2807465, 183.0075378, -331.4919128, 344.3281250
1: -116.2964935, 126.5755463, -165.1631775, 173.1105804, -289.4070740, 291.7387085
2: -168.2982788, 140.5842438, -238.5625763, 190.9392853, -359.2375488, 379.1468201
3: -68.7259674, 169.2932587, -94.2064362, 236.2539673, -304.9798584, 262.9326477
4: -187.5333405, 139.8242035, -265.5162659, 190.8201752, -378.3535156, 405.3404541

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246764, upper bound: 187.4220328
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301238, upper bound: 187.6301412
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -90.0296860, 95.7459259, -93.5899734, 92.8750916, -182.9047852, 189.3359070
1: -70.5223999, 90.7102203, -73.1107025, 87.0040359, -157.5264130, 163.8209076
2: -102.4902802, 100.3906708, -105.9443512, 97.7353897, -200.2256775, 206.3350067
3: -48.7527466, 108.2179565, -47.2811356, 109.5639648, -158.3167114, 155.4990845
4: -114.9058533, 99.2772675, -118.2672424, 97.3865814, -212.2924347, 217.5445099

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8198218, upper bound: 187.4231113
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8218635, upper bound: 187.6289039
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -143.9209900, 137.5663147, -93.5899734, 92.8750916, -236.7960815, 230.3369598
1: -112.8730698, 130.6749878, -73.1107025, 87.0040359, -199.8771057, 202.0905457
2: -163.7390137, 143.6699066, -105.9443512, 97.7353897, -261.4743958, 247.6691284
3: -70.0665512, 166.9166107, -47.2811356, 109.5639648, -177.7084351, 214.1977539
4: -182.9464722, 142.4316864, -118.2672424, 97.3865814, -280.3330383, 259.8267212

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8198218, upper bound: 187.4231113
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8218635, upper bound: 187.6289039
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -90.0296860, 95.7459259, -149.9745941, 137.3406982, -227.3703766, 245.7205200
1: -70.5223999, 90.7102203, -117.3068619, 129.7829132, -200.3052979, 208.0170746
2: -102.4902802, 100.3906708, -169.9014282, 144.0322723, -246.5225220, 270.2921143
3: -48.7527466, 108.2179565, -70.6789017, 171.0912170, -219.8439636, 178.8968506
4: -114.9058533, 99.2772675, -189.4996490, 143.3141022, -258.2199402, 288.7769165

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269706, upper bound: 187.4231354
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290123, upper bound: 187.6289272
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -143.9209900, 137.5663147, -149.9745941, 137.3406982, -281.2616882, 287.0580444
1: -112.8730698, 130.6749878, -117.3068619, 129.7829132, -242.6559753, 246.5460815
2: -163.7390137, 143.6699066, -169.9014282, 144.0322723, -307.7713013, 311.9698181
3: -70.0665512, 166.9166107, -70.6789017, 171.0912170, -239.3860626, 237.5955048
4: -182.9464722, 142.4316864, -189.4996490, 143.3141022, -326.2605591, 331.5132751

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269706, upper bound: 187.4231354
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290123, upper bound: 187.6289272
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -93.5899734, 92.8750916, -239.3872375, 228.6894531
1: -114.6307831, 127.5979538, -73.1107025, 87.0040359, -201.6348267, 200.7086487
2: -166.0353241, 141.6752014, -105.9443512, 97.7353897, -263.7707214, 247.6194916
3: -69.5864105, 167.3457031, -47.2811356, 109.5639648, -179.1503754, 214.6268311
4: -185.1463776, 140.9630737, -118.2672424, 97.3865814, -282.5329285, 259.2302551

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446981, upper bound: 187.6245879
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209895, upper bound: 187.6308760
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -93.5899734, 92.8750916, -303.1558228, 276.5975037
1: -165.1631775, 173.1105804, -73.1107025, 87.0040359, -252.1672058, 246.2212830
2: -238.5625763, 190.9392853, -105.9443512, 97.7353897, -336.2979736, 296.8836365
3: -94.2064362, 236.2539673, -47.2811356, 109.5639648, -202.9987946, 283.5350952
4: -265.5162659, 190.8201752, -118.2672424, 97.3865814, -362.9028015, 309.0873718

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446981, upper bound: 187.6245879
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209895, upper bound: 187.6308760
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -149.9745941, 137.3406982, -283.8528442, 285.0740967
1: -114.6307831, 127.5979538, -117.3068619, 129.7829132, -244.4136658, 244.9048157
2: -166.0353241, 141.6752014, -169.9014282, 144.0322723, -310.0675354, 311.5765991
3: -69.5864105, 167.3457031, -70.6789017, 171.0912170, -240.6776123, 238.0245972
4: -185.1463776, 140.9630737, -189.4996490, 143.3141022, -328.4604797, 330.4627075

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6285038
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6292015
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -149.9745941, 137.3406982, -347.6214600, 332.9821167
1: -165.1631775, 173.1105804, -117.3068619, 129.7829132, -294.9461060, 290.4174500
2: -238.5625763, 190.9392853, -169.9014282, 144.0322723, -382.5947876, 360.8406982
3: -94.2064362, 236.2539673, -70.6789017, 171.0912170, -264.6764832, 306.9328308
4: -265.5162659, 190.8201752, -189.4996490, 143.3141022, -408.8303833, 380.3197937

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6285038
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6292015
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -90.0296860, 95.7459259, -149.2839355, 134.6893616, -224.7190399, 245.0298462
1: -70.5223999, 90.7102203, -116.9905624, 127.1221771, -197.6445618, 207.7007751
2: -102.4902802, 100.3906708, -169.2326202, 141.1778259, -243.6681061, 269.6232910
3: -48.7527466, 108.2179565, -69.0598907, 170.1712341, -218.9239655, 177.2778320
4: -114.9058533, 99.2772675, -188.5170746, 140.5437622, -255.4495544, 287.7943420

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258287, upper bound: 187.4231287
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278704, upper bound: 187.6289213
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -143.9209900, 137.5663147, -149.2839355, 134.6893616, -278.6102600, 286.4306641
1: -112.8730698, 130.6749878, -116.9905624, 127.1221771, -239.9952240, 246.2777710
2: -163.7390137, 143.6699066, -169.2326202, 141.1778259, -304.9168396, 311.3722839
3: -70.0665512, 166.9166107, -69.0598907, 170.1712341, -238.5224304, 235.9764709
4: -182.9464722, 142.4316864, -188.5170746, 140.5437622, -323.4902344, 330.6036682

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258287, upper bound: 187.4231113
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278704, upper bound: 187.6289039
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -90.0296860, 95.7459259, -214.7231140, 185.9732056, -276.0028992, 310.4690247
1: -70.5223999, 90.7102203, -168.6910553, 175.9269562, -246.4493561, 259.4012146
2: -102.4902802, 100.3906708, -243.5518188, 193.9766388, -296.4669189, 343.9425049
3: -48.7527466, 108.2179565, -95.6573029, 241.0182190, -289.7709656, 203.1670074
4: -114.9058533, 99.2772675, -271.0773621, 193.9729767, -308.8787537, 370.3546143

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6257908
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6257908
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -143.9209900, 137.5663147, -214.7231140, 185.9732056, -329.8941956, 352.1930847
1: -112.8730698, 130.6749878, -168.6910553, 175.9269562, -288.8000183, 298.1291199
2: -163.7390137, 143.6699066, -243.5518188, 193.9766388, -357.7156372, 385.9393005
3: -70.0665512, 166.9166107, -95.6573029, 241.0182190, -309.4717407, 262.0506897
4: -182.9464722, 142.4316864, -271.0773621, 193.9729767, -376.9194336, 413.5090027

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6257908
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6289272
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -149.2839355, 134.6893616, -281.2014465, 284.3834229
1: -114.6307831, 127.5979538, -116.9905624, 127.1221771, -241.7529449, 244.5885010
2: -166.0353241, 141.6752014, -169.2326202, 141.1778259, -307.2131348, 310.9078064
3: -69.5864105, 167.3457031, -69.0598907, 170.1712341, -239.7576141, 236.4055786
4: -185.1463776, 140.9630737, -188.5170746, 140.5437622, -325.6901245, 329.4801331

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295757, upper bound: 187.6276533
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295757, upper bound: 187.6284155
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -149.2839355, 134.6893616, -344.9700928, 332.2914429
1: -165.1631775, 173.1105804, -116.9905624, 127.1221771, -292.2853088, 290.1011353
2: -238.5625763, 190.9392853, -169.2326202, 141.1778259, -379.7403564, 360.1719055
3: -94.2064362, 236.2539673, -69.0598907, 170.1712341, -263.8127441, 305.3138428
4: -265.5162659, 190.8201752, -188.5170746, 140.5437622, -406.0600281, 379.3372498

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4174430, upper bound: 187.6245879
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301412, upper bound: 187.6308760
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -214.7231140, 185.9732056, -332.4853516, 349.8226013
1: -114.6307831, 127.5979538, -168.6910553, 175.9269562, -290.5577087, 296.2889404
2: -166.0353241, 141.6752014, -243.5518188, 193.9766388, -360.0119629, 385.2269592
3: -69.5864105, 167.3457031, -95.6573029, 241.0182190, -310.6046143, 262.1953735
4: -185.1463776, 140.9630737, -271.0773621, 193.9729767, -379.1193237, 412.0404358

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6278615
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6286236
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -214.7231140, 185.9732056, -396.2539673, 397.7306213
1: -165.1631775, 173.1105804, -168.6910553, 175.9269562, -341.0900574, 341.8016052
2: -238.5625763, 190.9392853, -243.5518188, 193.9766388, -432.5392151, 434.4910889
3: -94.2064362, 236.2539673, -95.6573029, 241.0182190, -334.7620850, 331.2743225
4: -265.5162659, 190.8201752, -271.0773621, 193.9729767, -459.4892578, 461.8974915

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6278615
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6286236
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.86 seconds
NS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8151774, upper bound: 187.3663697
NS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8144769, upper bound: 187.5068514
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.3509328, upper bound: 187.4854176
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8205051, upper bound: 187.5295517
NS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.4446681, upper bound: 187.6238230
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8209595, upper bound: 187.6301112
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.4446681, upper bound: 187.6238230
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8209595, upper bound: 187.6301112
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6279076
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6287846
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6279076
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6287846
NS_A1_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.4854176, upper bound: 187.3509328
NS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5295517, upper bound: 187.8205051
NS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5249229, upper bound: 187.7697195
NS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5295517, upper bound: 187.8122145
NS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.3663784, upper bound: 187.6259274
NS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5068601, upper bound: 187.6252269
NS_A1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5273276
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5263085, upper bound: 187.6279076
NS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6238230, upper bound: 187.4446807
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6301112, upper bound: 187.8209721
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6238230, upper bound: 187.4446807
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6301112, upper bound: 187.8209721
NS_A1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5295517
NS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5263085, upper bound: 187.6279424
NS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6246764, upper bound: 187.4220328
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6301238, upper bound: 187.6301412
NS_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8198218, upper bound: 187.4231113
NS_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8218635, upper bound: 187.6289039
NS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8198218, upper bound: 187.4231113
NS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8218635, upper bound: 187.6289039
NS_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6269706, upper bound: 187.4231354
NS_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6290123, upper bound: 187.6289272
NS_A2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6269706, upper bound: 187.4231354
NS_A2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6290123, upper bound: 187.6289272
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.4446981, upper bound: 187.6245879
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8209895, upper bound: 187.6308760
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.4446981, upper bound: 187.6245879
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.8209895, upper bound: 187.6308760
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6285038
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6292015
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6285038
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6292015
NS_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6258287, upper bound: 187.4231287
NS_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6278704, upper bound: 187.6289213
NS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6258287, upper bound: 187.4231113
NS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6278704, upper bound: 187.6289039
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6257908
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6257908
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6257908
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6289272
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5295757, upper bound: 187.6276533
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.5295757, upper bound: 187.6284155
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.4174430, upper bound: 187.6245879
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6301412, upper bound: 187.6308760
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6278615
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6286236
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6278615
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.86
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6286236

## BFS NS instance: NS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -93.5899734, 92.8750916, -127.1413193, 144.4837952
1: -27.0025425, 47.7687073, -73.1107025, 87.0040359, -114.0065613, 120.8794022
2: -39.5765114, 53.9266510, -105.9443512, 97.7353897, -137.3118896, 159.8710022
3: -25.4885406, 48.3789978, -47.2811356, 109.5639648, -135.0525055, 95.6601334
4: -44.9341927, 53.1512604, -118.2672424, 97.3865814, -142.3207550, 171.4184570

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8159881, upper bound: 187.3877039
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8159881, upper bound: 187.3900516
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -39.0077286, 53.3303146, -93.5899734, 92.8750916, -131.8828125, 146.9202881
1: -30.5722923, 50.0091782, -73.1107025, 87.0040359, -117.5763245, 123.1198654
2: -44.8006439, 56.3047638, -105.9443512, 97.7353897, -142.5360413, 162.2491150
3: -26.5797043, 52.9133911, -47.2811356, 109.5639648, -136.1436768, 100.1945267
4: -50.5919037, 55.6129456, -118.2672424, 97.3865814, -147.9784851, 173.8801880

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7739850, upper bound: 187.7875841
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7443952, upper bound: 187.7588875
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -87.3655396, 92.1814575, -93.4733276, 92.8134308, -180.1789551, 185.6547546
1: -68.5108109, 87.2906647, -73.0208511, 86.9460297, -155.4568329, 160.3114929
2: -99.3974304, 96.7410889, -105.8147049, 97.6726913, -197.0701294, 202.5557556
3: -46.8131294, 105.1580734, -47.2495461, 109.4501266, -156.2632446, 152.4076080
4: -111.3977280, 95.6210403, -118.1232986, 97.3208160, -208.7185364, 213.7443390

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149339, upper bound: 187.3663697
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8142334, upper bound: 187.5068514
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -149.9745941, 137.3406982, -171.6068878, 200.8684387
1: -27.0025425, 47.7687073, -117.3068619, 129.7829132, -156.7854614, 165.0755615
2: -39.5765114, 53.9266510, -169.9014282, 144.0322723, -183.6087036, 223.8280640
3: -25.4885406, 48.3789978, -70.6789017, 171.0912170, -196.5797577, 119.0578995
4: -44.9341927, 53.1512604, -189.4996490, 143.3141022, -188.2482910, 242.6509094

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3878187
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3900510
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -39.0077286, 53.3303146, -149.9745941, 137.3406982, -176.3484039, 203.3049011
1: -30.5722923, 50.0091782, -117.3068619, 129.7829132, -160.3552094, 167.3160248
2: -44.8006439, 56.3047638, -169.9014282, 144.0322723, -188.8328857, 226.2061920
3: -26.5797043, 52.9133911, -70.6789017, 171.0912170, -197.6709290, 123.5922928
4: -50.5919037, 55.6129456, -189.4996490, 143.3141022, -193.9060059, 245.1125946

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8150484
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8172807
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -78.6385880, 87.7329025, -149.9745941, 137.3406982, -215.9792786, 237.7074890
1: -61.8511772, 83.0646057, -117.3068619, 129.7829132, -191.6340942, 200.3714600
2: -89.8312149, 92.2100830, -169.9014282, 144.0322723, -233.8634338, 262.1115112
3: -44.6076813, 96.7816925, -70.6789017, 171.0912170, -215.6988983, 167.4605865
4: -100.9121475, 90.8883057, -189.4996490, 143.3141022, -244.2262573, 280.3879395

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3641368
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3663691
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -84.4544296, 90.4469299, -149.9745941, 137.3406982, -221.7950897, 240.4215240
1: -66.2400513, 85.6331940, -117.3068619, 129.7829132, -196.0229492, 202.9400635
2: -96.1552505, 94.9304047, -169.9014282, 144.0322723, -240.1874390, 264.8318481
3: -45.9472961, 102.2089539, -70.6789017, 171.0912170, -217.0385132, 172.8878479
4: -107.8291397, 93.7583084, -189.4996490, 143.3141022, -251.1432190, 283.2579651

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5046185
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5068508
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -78.3078232, 83.3920822, -172.8399963, 168.1897278
1: -69.8412399, 84.1715393, -61.0082550, 78.1903000, -148.0315399, 145.1797791
2: -101.2739258, 94.6007919, -88.5467072, 88.3264847, -189.6004028, 183.1474609
3: -45.8811989, 105.0105896, -42.9643326, 92.9597931, -138.8409882, 147.9749146
4: -113.0568390, 94.2403870, -99.0556564, 87.5924988, -200.6492767, 193.2960510

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.2425294, upper bound: 187.7207162
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4421850, upper bound: 187.7255157
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -93.0833359, 92.5407257, -181.9887085, 182.9652405
1: -69.8412399, 84.1715393, -72.7047272, 86.7016907, -156.5429382, 156.8762207
2: -101.2739258, 94.6007919, -105.3617554, 97.4237671, -198.6976471, 199.9625549
3: -45.8811989, 105.0105896, -47.1332130, 109.0123520, -154.8935547, 152.1437988
4: -113.0568390, 94.2403870, -117.6394958, 97.0536499, -210.1104889, 211.8798828

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209595, upper bound: 187.8185629
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209595, upper bound: 187.8185629
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -78.3078232, 83.3920822, -227.6588440, 209.5378113
1: -112.9824219, 123.8839874, -61.0082550, 78.1903000, -191.1727295, 184.8922424
2: -163.5588074, 137.6753693, -88.5467072, 88.3264847, -251.8852844, 226.2220459
3: -67.3619995, 164.7313690, -42.9643326, 92.9597931, -160.3217926, 207.6956940
4: -182.2491913, 136.8414307, -99.0556564, 87.5924988, -269.8416748, 235.8970947

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4444952, upper bound: 187.6232985
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4425973, upper bound: 187.5374617
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -93.0833359, 92.5407257, -236.8075104, 224.3133240
1: -112.9824219, 123.8839874, -72.7047272, 86.7016907, -199.6841125, 196.5887146
2: -163.5588074, 137.6753693, -105.3617554, 97.4237671, -260.9825439, 243.0371094
3: -67.3619995, 164.7313690, -47.1332130, 109.0123520, -176.3743591, 211.8645782
4: -182.2491913, 136.8414307, -117.6394958, 97.0536499, -279.3028259, 254.4809265

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7321553, upper bound: 187.6276281
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7302574, upper bound: 187.5417912
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -90.0296860, 95.7459259, -185.1938782, 179.9115906
1: -69.8412399, 84.1715393, -70.5223999, 90.7102203, -160.5514374, 154.6938934
2: -101.2739258, 94.6007919, -102.4902802, 100.3906708, -201.6645660, 197.0910645
3: -45.8811989, 105.0105896, -48.7527466, 108.2179565, -154.0991516, 153.6334076
4: -113.0568390, 94.2403870, -114.9058533, 99.2772675, -212.3341064, 209.1462402

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6081341, upper bound: 187.8064415
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8132338
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -146.0465698, 134.7337646, -224.1817169, 235.9284668
1: -69.8412399, 84.1715393, -114.2232513, 127.2852173, -197.1264496, 198.3947754
2: -101.2739258, 94.6007919, -165.4884644, 141.3374176, -242.6113129, 260.0892639
3: -45.8811989, 105.0105896, -69.3965378, 166.8359833, -212.7171631, 174.4071350
4: -113.0568390, 94.2403870, -184.5755615, 140.5543518, -253.6111755, 278.8159485

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271923, upper bound: 187.7362092
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277906, upper bound: 187.6880296
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283330, upper bound: 187.8217379
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -90.0296860, 95.7459259, -240.0127106, 221.2596588
1: -112.9824219, 123.8839874, -70.5223999, 90.7102203, -203.6926422, 194.4063873
2: -163.5588074, 137.6753693, -102.4902802, 100.3906708, -263.9494629, 240.1656494
3: -67.3619995, 164.7313690, -48.7527466, 108.2179565, -175.5799255, 213.4841003
4: -182.2491913, 136.8414307, -114.9058533, 99.2772675, -281.5264587, 251.7472229

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3743426, upper bound: 187.6269345
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263123, upper bound: 187.6257991
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -146.0465698, 134.7337646, -279.0005493, 277.2765198
1: -112.9824219, 123.8839874, -114.2232513, 127.2852173, -240.2676239, 238.1072388
2: -163.5588074, 137.6753693, -165.4884644, 141.3374176, -304.8962402, 303.1638184
3: -67.3619995, 164.7313690, -69.3965378, 166.8359833, -234.1979675, 234.1278992
4: -182.2491913, 136.8414307, -184.5755615, 140.5543518, -322.8035278, 321.4169922

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4680153
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6264775
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -93.4733276, 92.8134308, -88.6077271, 92.8103714, -186.2836609, 181.4211273
1: -73.0208511, 86.9460297, -69.5185242, 87.8645935, -160.8854370, 156.4645538
2: -105.8147049, 97.6726913, -100.9003143, 97.3328018, -203.1474762, 198.5729980
3: -47.2495461, 109.4501266, -47.0367928, 106.5991135, -153.8486481, 156.4869232
4: -118.1232986, 97.3208160, -113.0341187, 96.2283783, -214.3516693, 210.3549042

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663697, upper bound: 187.8149339
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068514, upper bound: 187.8142334
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -90.7469177, 90.6126022, -143.9209900, 137.5663147, -227.4749756, 234.5335693
1: -70.7987976, 84.9603500, -112.8730698, 130.6749878, -199.7688904, 197.8334198
2: -102.6348648, 95.4938202, -163.7390137, 143.6699066, -244.3314667, 259.2328491
3: -46.0043831, 106.2343369, -70.0665512, 166.9166107, -212.9209900, 174.4003143
4: -114.5984802, 95.0162354, -182.9464722, 142.4316864, -256.1369324, 277.9627075

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045601, upper bound: 187.7649222
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045601, upper bound: 187.7740562
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -92.5626678, 92.1819839, -143.9209900, 137.5663147, -229.2443390, 236.1029663
1: -72.2883682, 86.3782349, -112.8730698, 130.6749878, -201.2286835, 199.2513123
2: -104.7752304, 97.0700226, -163.7390137, 143.6699066, -246.4468842, 260.8090210
3: -46.9766197, 108.4464035, -70.0665512, 166.9166107, -213.8932343, 176.5791016
4: -116.9716492, 96.6647339, -182.9464722, 142.4316864, -258.4617615, 279.6112061

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4880256
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260375, upper bound: 187.8100097
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -79.2864609, 88.0104218, -236.4947968, 213.3338623
1: -116.2964935, 126.5755463, -62.3454056, 83.3233795, -199.6198730, 188.9209442
2: -168.2982788, 140.5842438, -90.5679855, 92.4689178, -260.7672119, 231.1521912
3: -68.7259674, 169.2932587, -44.6927567, 97.4578094, -166.1837769, 213.9860077
4: -187.5333405, 139.8242035, -101.7387695, 91.1533737, -278.6866760, 241.5629425

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3635645, upper bound: 187.5047471
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3635645, upper bound: 187.6259274
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -148.4843750, 134.0473938, -85.4159851, 90.9214935, -239.4058685, 219.4633789
1: -116.2964935, 126.5755463, -67.0133209, 86.0685425, -202.3650360, 193.5888672
2: -168.2982788, 140.5842438, -97.3060455, 95.3715820, -263.6698608, 237.8902740
3: -68.7259674, 169.2932587, -46.1031952, 103.3089905, -172.0349579, 215.3964539
4: -187.5333405, 139.8242035, -109.0886841, 94.2118530, -281.7451477, 248.9128723

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5040462, upper bound: 187.5040466
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5040462, upper bound: 187.6252269
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -143.9209900, 137.5663147, -281.1489563, 275.1509705
1: -112.9824219, 123.8839874, -112.8730698, 130.6749878, -242.0775146, 236.7570496
2: -163.5588074, 137.6753693, -163.7390137, 143.6699066, -305.3424683, 301.4143677
3: -67.3619995, 164.7313690, -70.0665512, 166.9166107, -234.2785950, 232.8464050
4: -182.2491913, 136.8414307, -182.9464722, 142.4316864, -323.9643250, 319.7879028

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6059456, upper bound: 187.6218485
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.6279076
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -78.3078232, 83.3920822, -144.8092651, 131.6590424, -209.9668579, 228.2013245
1: -61.0082550, 78.1903000, -113.4549103, 124.2506790, -185.2589417, 191.6452026
2: -88.5467072, 88.3264847, -164.1941833, 138.0722961, -226.6189575, 252.5206604
3: -42.9643326, 92.9597931, -67.5839310, 165.3264618, -208.2908020, 160.5437317
4: -99.0556564, 87.5924988, -182.9160309, 137.3219604, -236.3776245, 270.5085449

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232985, upper bound: 187.4444952
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5374617, upper bound: 187.4425973
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -93.0833359, 92.5407257, -144.8092651, 131.6590424, -224.7423706, 237.3499908
1: -72.7047272, 86.7016907, -113.4549103, 124.2506790, -196.9554138, 200.1566010
2: -105.3617554, 97.4237671, -164.1941833, 138.0722961, -243.4340210, 261.6179504
3: -47.1332130, 109.0123520, -67.5839310, 165.3264618, -212.4596710, 176.5962830
4: -117.6394958, 97.0536499, -182.9160309, 137.3219604, -254.9614563, 279.9696655

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276281, upper bound: 187.7321553
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5417912, upper bound: 187.7302574
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -78.3078232, 83.3920822, -210.2807465, 183.0075378, -261.3153687, 293.6728210
1: -61.0082550, 78.1903000, -165.1631775, 173.1105804, -234.1188354, 243.3534851
2: -88.5467072, 88.3264847, -238.5625763, 190.9392853, -279.4859619, 326.8890381
3: -42.9643326, 92.9597931, -94.2064362, 236.2539673, -279.2182922, 186.2650757
4: -99.0556564, 87.5924988, -265.5162659, 190.8201752, -289.8758240, 353.1087646

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240634, upper bound: 187.4526768
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5384040, upper bound: 187.4465258
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -93.0833359, 92.5407257, -210.2807465, 183.0075378, -276.0908813, 302.8214722
1: -72.7047272, 86.7016907, -165.1631775, 173.1105804, -245.8152924, 251.8648224
2: -105.3617554, 97.4237671, -238.5625763, 190.9392853, -296.3010254, 335.9863281
3: -47.1332130, 109.0123520, -94.2064362, 236.2539673, -283.3871155, 202.4570770
4: -117.6394958, 97.0536499, -265.5162659, 190.8201752, -308.4596558, 362.5699158

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283930, upper bound: 187.7403369
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5427335, upper bound: 187.7341859
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -144.8092651, 131.6590424, -275.9258423, 276.0392151
1: -112.9824219, 123.8839874, -113.4549103, 124.2506790, -237.2330933, 237.3388977
2: -163.5588074, 137.6753693, -164.1941833, 138.0722961, -301.6311035, 301.8695679
3: -67.3619995, 164.7313690, -67.5839310, 165.3264618, -232.6884613, 232.3153076
4: -182.2491913, 136.8414307, -182.9160309, 137.3219604, -319.5711060, 319.7574463

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5047471, upper bound: 187.4680245
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5040466, upper bound: 187.6259756
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -131.2125702, 123.7671967, -210.2807465, 183.0075378, -314.2200928, 334.0479431
1: -102.6090622, 116.8737335, -165.1631775, 173.1105804, -275.7196350, 282.0368958
2: -148.5732117, 130.2702484, -238.5625763, 190.9392853, -339.5125122, 368.8327637
3: -63.8431320, 150.6193237, -94.2064362, 236.2539673, -300.0971069, 244.1388855
4: -165.7654877, 129.0743713, -265.5162659, 190.8201752, -356.5856628, 394.5906372

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244607, upper bound: 187.3312492
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6216478, upper bound: 187.4188230
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -147.9475708, 133.7212982, -210.2807465, 183.0075378, -330.9551086, 344.0020447
1: -115.8660278, 126.2766418, -165.1631775, 173.1105804, -288.9766235, 291.4397583
2: -167.6802826, 140.2731628, -238.5625763, 190.9392853, -358.6195679, 378.8357544
3: -68.5721893, 168.7137146, -94.2064362, 236.2539673, -304.8261414, 262.3620300
4: -186.8633575, 139.4981537, -265.5162659, 190.8201752, -377.6835327, 405.0144043

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5295612
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.6287846
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -75.1740952, 86.4655151, -93.5899734, 92.8750916, -168.0491943, 180.0554810
1: -59.0501709, 81.8904648, -73.1107025, 87.0040359, -146.0541992, 155.0011597
2: -86.0124969, 90.7257843, -105.9443512, 97.7353897, -183.7478943, 196.6701355
3: -43.9294243, 93.0511627, -47.2811356, 109.5639648, -153.4586639, 140.3323059
4: -96.5924683, 89.4440918, -118.2672424, 97.3865814, -193.9790344, 207.7112885

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8139011, upper bound: 186.8652443
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8135479, upper bound: 187.4303898
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -89.8806458, 95.6513596, -93.5899734, 92.8750916, -182.7557373, 189.2413330
1: -70.4029846, 90.6203918, -73.1107025, 87.0040359, -157.4070129, 163.7310791
2: -102.3191757, 100.2926941, -105.9443512, 97.7353897, -200.0545349, 206.2369995
3: -48.7057953, 108.0552597, -47.2811356, 109.5639648, -158.2697601, 155.3363953
4: -114.7167358, 99.1772308, -118.2672424, 97.3865814, -212.1033173, 217.4444275

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8161505, upper bound: 187.3743339
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8151150, upper bound: 187.6263037
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -126.8857346, 127.0289917, -93.5899734, 92.8750916, -219.7608337, 218.7695465
1: -99.5447693, 120.7034454, -73.1107025, 87.0040359, -186.5487976, 191.2532043
2: -144.3884735, 132.6780396, -105.9443512, 97.7353897, -242.1238708, 235.5974731
3: -64.6005859, 148.9533081, -47.2811356, 109.5639648, -171.8814240, 196.2344360
4: -161.6483459, 131.3171997, -118.2672424, 97.3865814, -259.0349121, 247.6131744

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7943456, upper bound: 187.3931694
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8161543, upper bound: 186.8935726
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8154016, upper bound: 187.4195648
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -143.7625427, 137.4667053, -93.5899734, 92.8750916, -236.6376343, 230.1980591
1: -112.7459106, 130.5810547, -73.1107025, 87.0040359, -199.7499390, 201.9553223
2: -163.5565948, 143.5669098, -105.9443512, 97.7353897, -261.2919922, 247.5246124
3: -70.0163727, 166.7445679, -47.2811356, 109.5639648, -177.6428680, 214.0256958
4: -182.7444916, 142.3251953, -118.2672424, 97.3865814, -280.1310425, 259.6799927

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7740526, upper bound: 187.6242752
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8165476, upper bound: 187.6289039
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -75.1740952, 86.4655151, -149.9745941, 137.3406982, -212.5148010, 236.4401093
1: -59.0501709, 81.8904648, -117.3068619, 129.7829132, -188.8330688, 199.1973114
2: -86.0124969, 90.7257843, -169.9014282, 144.0322723, -230.0447388, 260.6271973
3: -43.9294243, 93.0511627, -70.6789017, 171.0912170, -215.0206451, 163.7300720
4: -96.5924683, 89.4440918, -189.4996490, 143.3141022, -239.9065552, 278.9437256

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257641, upper bound: 186.8652439
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254109, upper bound: 187.4303892
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -89.8806458, 95.6513596, -149.9745941, 137.3406982, -227.2213287, 245.6259460
1: -70.4029846, 90.6203918, -117.3068619, 129.7829132, -200.1858978, 207.9272461
2: -102.3191757, 100.2926941, -169.9014282, 144.0322723, -246.3514099, 270.1941223
3: -48.7057953, 108.0552597, -70.6789017, 171.0912170, -219.7970123, 178.7341614
4: -114.7167358, 99.1772308, -189.4996490, 143.3141022, -258.0308228, 288.6768494

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254956, upper bound: 187.3743333
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269780, upper bound: 187.6263031
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -126.8857346, 127.0289917, -149.9745941, 137.3406982, -264.2264404, 275.4906311
1: -99.5447693, 120.7034454, -117.3068619, 129.7829132, -229.3276672, 235.7087250
2: -144.3884735, 132.6780396, -169.9014282, 144.0322723, -288.4207458, 299.8981018
3: -64.6005859, 148.9533081, -70.6789017, 171.0912170, -233.5590668, 219.6322021
4: -161.6483459, 131.3171997, -189.4996490, 143.3141022, -304.9624634, 319.2996826

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280174, upper bound: 186.8935722
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272646, upper bound: 187.4195642
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -143.7625427, 137.4667053, -149.9745941, 137.3406982, -281.1032410, 286.9192200
1: -112.7459106, 130.5810547, -117.3068619, 129.7829132, -242.5288239, 246.4108734
2: -163.5565948, 143.5669098, -169.9014282, 144.0322723, -307.5888062, 311.8252258
3: -70.0163727, 166.7445679, -70.6789017, 171.0912170, -239.3205414, 237.4234619
4: -182.7444916, 142.3251953, -189.4996490, 143.3141022, -326.0585938, 331.3665161

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302525, upper bound: 187.6264743
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302525, upper bound: 187.6289272
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -78.3078232, 83.3920822, -229.9042358, 213.4073181
1: -114.6307831, 127.5979538, -61.0082550, 78.1903000, -192.8210754, 188.6062012
2: -166.0353241, 141.6752014, -88.5467072, 88.3264847, -254.3617706, 230.2218628
3: -69.5864105, 167.3457031, -42.9643326, 92.9597931, -162.5461731, 210.3100128
4: -185.1463776, 140.9630737, -99.0556564, 87.5924988, -272.7388611, 240.0187378

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4445361, upper bound: 187.6244405
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3442679, upper bound: 187.6239750
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4399181, upper bound: 187.6213623
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -93.0833359, 92.5407257, -239.0528870, 228.1828308
1: -114.6307831, 127.5979538, -72.7047272, 86.7016907, -201.3324738, 200.3026733
2: -166.0353241, 141.6752014, -105.3617554, 97.4237671, -263.4590454, 247.0368958
3: -69.5864105, 167.3457031, -47.1332130, 109.0123520, -178.5987549, 214.4789124
4: -185.1463776, 140.9630737, -117.6394958, 97.0536499, -282.2000122, 258.6025696

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7321962, upper bound: 187.6287700
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6821948, upper bound: 187.6297634
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8201621, upper bound: 187.6304231
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -78.3078232, 83.3920822, -293.6728210, 261.3153687
1: -165.1631775, 173.1105804, -61.0082550, 78.1903000, -243.3534851, 234.1188354
2: -238.5625763, 190.9392853, -88.5467072, 88.3264847, -326.8890686, 279.4859619
3: -94.2064362, 236.2539673, -42.9643326, 92.9597931, -186.2650757, 279.2182922
4: -265.5162659, 190.8201752, -99.0556564, 87.5924988, -353.1087646, 289.8758240

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4526768, upper bound: 187.6240634
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4465258, upper bound: 187.5384040
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -93.0833359, 92.5407257, -302.8214722, 276.0908813
1: -165.1631775, 173.1105804, -72.7047272, 86.7016907, -251.8648529, 245.8153076
2: -238.5625763, 190.9392853, -105.3617554, 97.4237671, -335.9863281, 296.3010254
3: -94.2064362, 236.2539673, -47.1332130, 109.0123520, -202.4570618, 283.3871460
4: -265.5162659, 190.8201752, -117.6394958, 97.0536499, -362.5699158, 308.4596558

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7403369, upper bound: 187.6283930
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7341859, upper bound: 187.5427335
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -90.0296860, 95.7459259, -242.2580719, 225.1291809
1: -114.6307831, 127.5979538, -70.5223999, 90.7102203, -205.3410034, 198.1203308
2: -166.0353241, 141.6752014, -102.4902802, 100.3906708, -266.4259949, 244.1654663
3: -69.5864105, 167.3457031, -48.7527466, 108.2179565, -177.8043365, 216.0905762
4: -185.1463776, 140.9630737, -114.9058533, 99.2772675, -284.4236450, 255.8688965

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3741240, upper bound: 187.6278511
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263031, upper bound: 187.6267974
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -146.0465698, 134.7337646, -281.2458801, 281.1460571
1: -114.6307831, 127.5979538, -114.2232513, 127.2852173, -241.9160004, 241.8211975
2: -166.0353241, 141.6752014, -165.4884644, 141.3374176, -307.3727417, 307.1636658
3: -69.5864105, 167.3457031, -69.3965378, 166.8359833, -236.4223938, 236.7422485
4: -185.1463776, 140.9630737, -184.5755615, 140.5543518, -325.7007446, 325.5386353

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3741240, upper bound: 187.6286205
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263031, upper bound: 187.6273118
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -90.0296860, 95.7459259, -306.0266418, 273.0372314
1: -165.1631775, 173.1105804, -70.5223999, 90.7102203, -255.8733673, 243.6329498
2: -238.5625763, 190.9392853, -102.4902802, 100.3906708, -338.9532471, 293.4295349
3: -94.2064362, 236.2539673, -48.7527466, 108.2179565, -201.6632538, 285.0066833
4: -265.5162659, 190.8201752, -114.9058533, 99.2772675, -364.7935181, 305.7259521

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3791934, upper bound: 187.6275077
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311631, upper bound: 187.6264503
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -146.0465698, 134.7337646, -345.0144653, 329.0541077
1: -165.1631775, 173.1105804, -114.2232513, 127.2852173, -292.4483643, 287.3338318
2: -238.5625763, 190.9392853, -165.4884644, 141.3374176, -379.8999939, 356.4277344
3: -94.2064362, 236.2539673, -69.3965378, 166.8359833, -260.1801758, 305.6505127
4: -265.5162659, 190.8201752, -184.5755615, 140.5543518, -406.0706177, 375.3957520

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3791934, upper bound: 187.6282927
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311632, upper bound: 187.6269786
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -75.1740952, 86.4655151, -149.2839355, 134.6893616, -209.8634491, 235.7494507
1: -59.0501709, 81.8904648, -116.9905624, 127.1221771, -186.1723480, 198.8810272
2: -86.0124969, 90.7257843, -169.2326202, 141.1778259, -227.1903229, 259.9584045
3: -43.9294243, 93.0511627, -69.0598907, 170.1712341, -214.1006622, 162.1110382
4: -96.5924683, 89.4440918, -188.5170746, 140.5437622, -237.1361847, 277.9611816

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246511, upper bound: 186.8652529
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245638, upper bound: 187.4303984
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -89.8806458, 95.6513596, -149.2839355, 134.6893616, -224.5699921, 244.9353027
1: -70.4029846, 90.6203918, -116.9905624, 127.1221771, -197.5251617, 207.6109314
2: -102.3191757, 100.2926941, -169.2326202, 141.1778259, -243.4969635, 269.5253296
3: -48.7057953, 108.0552597, -69.0598907, 170.1712341, -218.8770142, 177.1151428
4: -114.7167358, 99.1772308, -188.5170746, 140.5437622, -255.2604980, 287.6943054

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5272904, upper bound: 187.6259194
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5272904, upper bound: 187.6291800
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -126.8857346, 127.0289917, -149.2839355, 134.6893616, -261.5750427, 274.8632507
1: -99.5447693, 120.7034454, -116.9905624, 127.1221771, -226.6669464, 235.4404144
2: -144.3884735, 132.6780396, -169.2326202, 141.1778259, -285.5662842, 299.3005981
3: -64.6005859, 148.9533081, -69.0598907, 170.1712341, -232.6954498, 218.0131836
4: -161.6483459, 131.3171997, -188.5170746, 140.5437622, -302.1920471, 318.3901062

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269043, upper bound: 186.8935813
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261516, upper bound: 187.4195648
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -143.7625427, 137.4667053, -149.2839355, 134.6893616, -278.4518433, 286.2917786
1: -112.7459106, 130.5810547, -116.9905624, 127.1221771, -239.8680878, 246.1425171
2: -163.5565948, 143.5669098, -169.2326202, 141.1778259, -304.7343750, 311.2277527
3: -70.0163727, 166.7445679, -69.0598907, 170.1712341, -238.4568939, 235.8044434
4: -182.7444916, 142.3251953, -188.5170746, 140.5437622, -323.2882690, 330.4569702

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306416, upper bound: 187.6256607
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306416, upper bound: 187.6289039
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -90.0296860, 95.7459259, -143.9574738, 137.5898285, -226.6428986, 239.7033997
1: -70.5223999, 90.7102203, -112.9016113, 130.6958466, -199.4725952, 203.6118011
2: -102.4902802, 100.3906708, -163.7802124, 143.6897125, -244.2701569, 264.1708984
3: -48.7527466, 108.2179565, -70.0736237, 166.9585724, -215.7112732, 176.3800354
4: -114.9058533, 99.2772675, -182.9930725, 142.4557343, -256.5001831, 282.2703247

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5249876, upper bound: 187.4305505
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266426, upper bound: 187.6259194
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -90.0296860, 95.7459259, -210.2807465, 183.0075378, -273.0372314, 306.0266724
1: -70.5223999, 90.7102203, -165.1631775, 173.1105804, -243.6329498, 255.8733215
2: -102.4902802, 100.3906708, -238.5625763, 190.9392853, -293.4295349, 338.9532471
3: -48.7527466, 108.2179565, -94.2064362, 236.2539673, -285.0066528, 201.6632385
4: -114.9058533, 99.2772675, -265.5162659, 190.8201752, -305.7260132, 364.7935181

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5249876, upper bound: 187.4305505
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266426, upper bound: 187.6291800
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -143.9209900, 137.5663147, -143.9574738, 137.5898285, -280.9091187, 280.9225464
1: -112.8730698, 130.6749878, -112.9016113, 130.6958466, -242.0233154, 242.0308380
2: -163.7390137, 143.6699066, -163.7802124, 143.6897125, -305.8538818, 305.8748169
3: -70.0665512, 166.9166107, -70.0736237, 166.9585724, -235.2983704, 235.2637177
4: -182.9464722, 142.4316864, -182.9930725, 142.4557343, -324.8988647, 324.9213562

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266377, upper bound: 187.4208500
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299938, upper bound: 187.6257908
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -143.9209900, 137.5663147, -210.2807465, 183.0075378, -326.9284973, 347.5751038
1: -112.8730698, 130.6749878, -165.1631775, 173.1105804, -285.9836426, 294.4980164
2: -163.7390137, 143.6699066, -238.5625763, 190.9392853, -354.6782837, 380.6659851
3: -70.0665512, 166.9166107, -94.2064362, 236.2539673, -304.4801636, 260.5468750
4: -182.9464722, 142.4316864, -265.5162659, 190.8201752, -373.7666626, 407.7019348

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266377, upper bound: 187.4231354
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299938, upper bound: 187.6289272
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -88.6077271, 92.8103714, -239.3225098, 223.7072144
1: -114.6307831, 127.5979538, -69.5185242, 87.8645935, -202.4953766, 197.1164856
2: -166.0353241, 141.6752014, -100.9003143, 97.3328018, -263.3681335, 242.5755157
3: -69.5864105, 167.3457031, -47.0367928, 106.5991135, -176.1855011, 214.3824921
4: -185.1463776, 140.9630737, -113.0341187, 96.2283783, -281.3746948, 253.9971924

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663691, upper bound: 187.6270404
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068508, upper bound: 187.6263399
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -144.8092651, 131.6590424, -278.1712036, 279.9087524
1: -114.6307831, 127.5979538, -113.4549103, 124.2506790, -238.8814697, 241.0528564
2: -166.0353241, 141.6752014, -164.1941833, 138.0722961, -304.1076050, 305.8693848
3: -69.5864105, 167.3457031, -67.5839310, 165.3264618, -234.9128723, 234.9296265
4: -185.1463776, 140.9630737, -182.9160309, 137.3219604, -322.4682312, 323.8790894

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663691, upper bound: 187.6279572
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068508, upper bound: 187.6269514
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -131.2125702, 123.7671967, -334.0479431, 314.2200928
1: -165.1631775, 173.1105804, -102.6090622, 116.8737335, -282.0368958, 275.7195740
2: -238.5625763, 190.9392853, -148.5732117, 130.2702484, -368.8327637, 339.5125122
3: -94.2064362, 236.2539673, -63.8431320, 150.6193237, -244.1388550, 300.0971069
4: -265.5162659, 190.8201752, -165.7654877, 129.0743713, -394.5906372, 356.5856628

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3354020, upper bound: 187.6236073
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4230128, upper bound: 187.6209946
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -148.6371918, 134.3037415, -344.5844727, 331.6447144
1: -165.1631775, 173.1105804, -116.4628754, 126.7690506, -291.9322205, 289.5734253
2: -238.5625763, 190.9392853, -168.4866028, 140.8123474, -379.3749084, 359.4259033
3: -94.2064362, 236.2539673, -68.8755569, 169.4759979, -263.1259155, 305.1294556
4: -265.5162659, 190.8201752, -187.7148285, 140.1523285, -405.6685791, 378.5350037

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276533
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6284155
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -143.9574738, 137.5898285, -283.3673706, 279.0569458
1: -114.6307831, 127.5979538, -112.9016113, 130.6958466, -243.7158203, 240.4995422
2: -166.0353241, 141.6752014, -163.7802124, 143.6897125, -307.7760010, 305.4553833
3: -69.5864105, 167.3457031, -70.0736237, 166.9585724, -236.5449524, 235.4083862
4: -185.1463776, 140.9630737, -182.9930725, 142.4557343, -326.8287964, 323.9561462

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6085106, upper bound: 187.5945368
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6280304
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -146.5121613, 135.0994873, -210.2807465, 183.0075378, -329.5196533, 345.3802490
1: -114.6307831, 127.5979538, -165.1631775, 173.1105804, -287.7413635, 292.7610779
2: -166.0353241, 141.6752014, -238.5625763, 190.9392853, -356.9745789, 380.2377319
3: -69.5864105, 167.3457031, -94.2064362, 236.2539673, -305.8403931, 260.6915894
4: -185.1463776, 140.9630737, -265.5162659, 190.8201752, -375.9665527, 406.4793396

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4348991
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260055, upper bound: 187.6269514
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -143.9574738, 137.5898285, -347.5988159, 326.9650269
1: -165.1631775, 173.1105804, -112.9016113, 130.6958466, -294.5191040, 286.0121460
2: -238.5625763, 190.9392853, -163.7802124, 143.6897125, -380.6861572, 354.7194824
3: -94.2064362, 236.2539673, -70.0736237, 166.9585724, -260.5887146, 304.4873657
4: -265.5162659, 190.8201752, -182.9930725, 142.4557343, -407.7260132, 373.8132324

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175638, upper bound: 187.6193874
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6278615
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -210.2807465, 183.0075378, -393.2882690, 393.2882690
1: -165.1631775, 173.1105804, -165.1631775, 173.1105804, -338.2737427, 338.2737427
2: -238.5625763, 190.9392853, -238.5625763, 190.9392853, -429.5018616, 429.5018311
3: -94.2064362, 236.2539673, -94.2064362, 236.2539673, -329.7705078, 329.7705383
4: -265.5162659, 190.8201752, -265.5162659, 190.8201752, -456.3364258, 456.3364258

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302750, upper bound: 187.4824702
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6286237
time: 0.55 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.91 seconds
NS_A1_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8159881, upper bound: 187.3877039
NS_A1_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8159881, upper bound: 187.3900516
NS_A1_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7739850, upper bound: 187.7875841
NS_A1_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7443952, upper bound: 187.7588875
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8149339, upper bound: 187.3663697
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8142334, upper bound: 187.5068514
NS_A1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3878187
NS_A1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3900510
NS_A1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8150484
NS_A1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8172807
NS_A1_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3641368
NS_A1_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3663691
NS_A1_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5046185
NS_A1_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5068508
NS_A1_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.2425294, upper bound: 187.7207162
NS_A1_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4421850, upper bound: 187.7255157
NS_A1_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8209595, upper bound: 187.8185629
NS_A1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8209595, upper bound: 187.8185629
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4444952, upper bound: 187.6232985
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4425973, upper bound: 187.5374617
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7321553, upper bound: 187.6276281
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7302574, upper bound: 187.5417912
NS_A1_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6081341, upper bound: 187.8064415
NS_A1_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8132338
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6277906, upper bound: 187.6880296
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6283330, upper bound: 187.8217379
NS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3743426, upper bound: 187.6269345
NS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6263123, upper bound: 187.6257991
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4680153
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6264775
NS_A1_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3663697, upper bound: 187.8149339
NS_A1_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5068514, upper bound: 187.8142334
NS_A1_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6045601, upper bound: 187.7649222
NS_A1_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6045601, upper bound: 187.7740562
NS_A1_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4880256
NS_A1_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6260375, upper bound: 187.8100097
NS_A1_B2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3635645, upper bound: 187.5047471
NS_A1_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3635645, upper bound: 187.6259274
NS_A1_B2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5040462, upper bound: 187.5040466
NS_A1_B2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5040462, upper bound: 187.6252269
NS_A1_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6059456, upper bound: 187.6218485
NS_A1_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6256607, upper bound: 187.6279076
NS_A1_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6232985, upper bound: 187.4444952
NS_A1_B2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5374617, upper bound: 187.4425973
NS_A1_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6276281, upper bound: 187.7321553
NS_A1_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5417912, upper bound: 187.7302574
NS_A1_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6240634, upper bound: 187.4526768
NS_A1_B2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5384040, upper bound: 187.4465258
NS_A1_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6283930, upper bound: 187.7403369
NS_A1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5427335, upper bound: 187.7341859
NS_A1_B2_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5047471, upper bound: 187.4680245
NS_A1_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5040466, upper bound: 187.6259756
NS_A1_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6244607, upper bound: 187.3312492
NS_A1_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6216478, upper bound: 187.4188230
NS_A1_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5295612
NS_A1_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6256607, upper bound: 187.6287846
NS_A2_B1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8139011, upper bound: 186.8652443
NS_A2_B1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8135479, upper bound: 187.4303898
NS_A2_B1_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8161505, upper bound: 187.3743339
NS_A2_B1_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8151150, upper bound: 187.6263037
NS_A2_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8161543, upper bound: 186.8935726
NS_A2_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8154016, upper bound: 187.4195648
NS_A2_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7740526, upper bound: 187.6242752
NS_A2_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8165476, upper bound: 187.6289039
NS_A2_B1_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6257641, upper bound: 186.8652439
NS_A2_B1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6254109, upper bound: 187.4303892
NS_A2_B1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6254956, upper bound: 187.3743333
NS_A2_B1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6269780, upper bound: 187.6263031
NS_A2_B1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6280174, upper bound: 186.8935722
NS_A2_B1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6272646, upper bound: 187.4195642
NS_A2_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6302525, upper bound: 187.6264743
NS_A2_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6302525, upper bound: 187.6289272
NS_A2_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3442679, upper bound: 187.6239750
NS_A2_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4399181, upper bound: 187.6213623
NS_A2_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6821948, upper bound: 187.6297634
NS_A2_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.8201621, upper bound: 187.6304231
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4526768, upper bound: 187.6240634
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4465258, upper bound: 187.5384040
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7403369, upper bound: 187.6283930
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.7341859, upper bound: 187.5427335
NS_A2_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3741240, upper bound: 187.6278511
NS_A2_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6263031, upper bound: 187.6267974
NS_A2_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3741240, upper bound: 187.6286205
NS_A2_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6263031, upper bound: 187.6273118
NS_A2_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3791934, upper bound: 187.6275077
NS_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6311631, upper bound: 187.6264503
NS_A2_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3791934, upper bound: 187.6282927
NS_A2_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6311632, upper bound: 187.6269786
NS_A2_B2_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6246511, upper bound: 186.8652529
NS_A2_B2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6245638, upper bound: 187.4303984
NS_A2_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5272904, upper bound: 187.6259194
NS_A2_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5272904, upper bound: 187.6291800
NS_A2_B2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6269043, upper bound: 186.8935813
NS_A2_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6261516, upper bound: 187.4195648
NS_A2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5306416, upper bound: 187.6256607
NS_A2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5306416, upper bound: 187.6289039
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5249876, upper bound: 187.4305505
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6266426, upper bound: 187.6259194
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5249876, upper bound: 187.4305505
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6266426, upper bound: 187.6291800
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6266377, upper bound: 187.4208500
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6299938, upper bound: 187.6257908
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6266377, upper bound: 187.4231354
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6299938, upper bound: 187.6289272
NS_A2_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3663691, upper bound: 187.6270404
NS_A2_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5068508, upper bound: 187.6263399
NS_A2_B2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3663691, upper bound: 187.6279572
NS_A2_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5068508, upper bound: 187.6269514
NS_A2_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.3354020, upper bound: 187.6236073
NS_A2_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.4230128, upper bound: 187.6209946
NS_A2_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276533
NS_A2_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6284155
NS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6085106, upper bound: 187.5945368
NS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6280304
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4348991
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6260055, upper bound: 187.6269514
NS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6175638, upper bound: 187.6193874
NS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6278615
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6302750, upper bound: 187.4824702
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.91
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6286237

## BFS NS instance: NS_A1_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -40.9809875, 54.7791138, -89.0453415, 91.8748169
1: -27.0025425, 47.7687073, -32.1188278, 51.3675804, -78.3701172, 79.8875198
2: -39.5765114, 53.9266510, -47.0364113, 57.8066292, -97.3831406, 100.9630585
3: -25.4885406, 48.3789978, -27.2729225, 55.0080185, -80.4965515, 75.6519165
4: -44.9341927, 53.1512604, -53.0424118, 57.1621284, -102.0963211, 106.1936646

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3887576, upper bound: 187.3877039
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3887576, upper bound: 187.3877039
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -89.4479752, 89.8819046, -124.1481323, 140.3417511
1: -27.0025425, 47.7687073, -69.8412399, 84.1715393, -111.1740723, 117.6099472
2: -39.5765114, 53.9266510, -101.2739258, 94.6007919, -134.1773071, 155.2005768
3: -25.4885406, 48.3789978, -45.8811989, 105.0105896, -130.4991302, 94.2601929
4: -44.9341927, 53.1512604, -113.0568390, 94.2403870, -139.1745758, 166.2080841

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3887576, upper bound: 187.3900516
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3887576, upper bound: 187.3900516
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -39.0077286, 53.3303146, -92.0074692, 91.8577042, -130.8654175, 145.3377686
1: -30.5722923, 50.0091782, -71.8683319, 86.0342560, -116.6065521, 121.8774643
2: -44.8006439, 56.3047638, -104.1582642, 96.6842041, -141.4848480, 160.4630280
3: -26.5797043, 52.9133911, -46.6869583, 107.9275589, -134.5072479, 99.6003494
4: -50.5919037, 55.6129456, -116.2792511, 96.3034668, -146.8953705, 171.8921967

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7621892, upper bound: 187.6605914
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7659606, upper bound: 187.7811411
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -39.0077286, 53.3303146, -99.8774261, 96.5714645, -135.5791931, 153.2077332
1: -30.5722923, 50.0091782, -78.0382080, 90.3266602, -120.8989487, 128.0473633
2: -44.8006439, 56.3047638, -113.1070404, 101.3708267, -146.1714783, 169.4118042
3: -26.5797043, 52.9133911, -48.8954735, 116.1568756, -142.7365570, 101.8088684
4: -50.5919037, 55.6129456, -126.1715927, 101.2557297, -151.8476257, 181.7845459

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7337585, upper bound: 187.6317045
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7375299, upper bound: 187.7522541
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -78.6385880, 87.7329025, -93.4733276, 92.8134308, -171.4519958, 181.2062225
1: -61.8511772, 83.0646057, -73.0208511, 86.9460297, -148.7971954, 156.0854492
2: -89.8312149, 92.2100830, -105.8147049, 97.6726913, -187.5039062, 198.0247803
3: -44.6076813, 96.7816925, -47.2495461, 109.4501266, -154.0578003, 144.0312195
4: -100.9121475, 90.8883057, -118.1232986, 97.3208160, -198.2329712, 209.0115967

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7848111, upper bound: 187.3375439
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149339, upper bound: 187.3640221
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149339, upper bound: 187.3663697
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -84.4544296, 90.4469299, -93.4733276, 92.8134308, -177.2678070, 183.9202576
1: -66.2400513, 85.6331940, -73.0208511, 86.9460297, -153.1860809, 158.6540527
2: -96.1552505, 94.9304047, -105.8147049, 97.6726913, -193.8279419, 200.7450867
3: -45.9472961, 102.2089539, -47.2495461, 109.4501266, -155.3974304, 149.4584656
4: -107.8291397, 93.7583084, -118.1232986, 97.3208160, -205.1499329, 211.8816071

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7840620, upper bound: 187.4780256
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4873317, upper bound: 187.5068518
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4873317, upper bound: 187.5068518
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -90.0296860, 95.7459259, -130.0121460, 140.9234924
1: -27.0025425, 47.7687073, -70.5223999, 90.7102203, -117.7127533, 118.2910995
2: -39.5765114, 53.9266510, -102.4902802, 100.3906708, -139.9671783, 156.4169312
3: -25.4885406, 48.3789978, -48.7527466, 108.2179565, -133.7064972, 96.9810333
4: -44.9341927, 53.1512604, -114.9058533, 99.2772675, -144.2114563, 168.0571136

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3728118, upper bound: 187.3878187
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3730399, upper bound: 187.3878187
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -146.0465698, 134.7337646, -169.0000000, 196.9403992
1: -27.0025425, 47.7687073, -114.2232513, 127.2852173, -154.2877655, 161.9919586
2: -39.5765114, 53.9266510, -165.4884644, 141.3374176, -180.9138794, 219.4151154
3: -25.4885406, 48.3789978, -69.3965378, 166.8359833, -192.3245239, 117.7755356
4: -44.9341927, 53.1512604, -184.5755615, 140.5543518, -185.4885101, 237.7268066

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6226543, upper bound: 186.8735962
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249775, upper bound: 187.3900510
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -39.0077286, 53.3303146, -90.0296860, 95.7459259, -134.7536469, 143.3600006
1: -30.5722923, 50.0091782, -70.5223999, 90.7102203, -121.2825165, 120.5315552
2: -44.8006439, 56.3047638, -102.4902802, 100.3906708, -145.1913147, 158.7950439
3: -26.5797043, 52.9133911, -48.7527466, 108.2179565, -134.7976532, 101.6393738
4: -50.5919037, 55.6129456, -114.9058533, 99.2772675, -149.8691711, 170.5187988

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5830068, upper bound: 187.7854177
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6188692, upper bound: 187.7644726
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8074614
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -39.0077286, 53.3303146, -146.0465698, 134.7337646, -173.7414856, 199.3768921
1: -30.5722923, 50.0091782, -114.2232513, 127.2852173, -157.8575134, 164.2323914
2: -44.8006439, 56.3047638, -165.4884644, 141.3374176, -186.1380463, 221.7932281
3: -26.5797043, 52.9133911, -69.3965378, 166.8359833, -193.4156799, 122.3099289
4: -50.5919037, 55.6129456, -184.5755615, 140.5543518, -191.1462402, 240.1885071

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5830068, upper bound: 187.7875836
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6225696, upper bound: 187.7709787
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239709, upper bound: 187.8172807
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -78.6385880, 87.7329025, -90.0296860, 95.7459259, -174.3845215, 177.7625885
1: -61.8511772, 83.0646057, -70.5223999, 90.7102203, -152.5613861, 153.5869751
2: -89.8312149, 92.2100830, -102.4902802, 100.3906708, -190.2218781, 194.7003632
3: -44.6076813, 96.7816925, -48.7527466, 108.2179565, -152.8256378, 145.5344238
4: -100.9121475, 90.8883057, -114.9058533, 99.2772675, -200.1894226, 205.7941589

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3722293, upper bound: 187.3641368
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3722293, upper bound: 187.3641368
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -78.6385880, 87.7329025, -146.0465698, 134.7337646, -213.3723450, 233.7794800
1: -61.8511772, 83.0646057, -114.2232513, 127.2852173, -189.1363983, 197.2878418
2: -89.8312149, 92.2100830, -165.4884644, 141.3374176, -231.1686096, 257.6985474
3: -44.6076813, 96.7816925, -69.3965378, 166.8359833, -211.4436646, 166.1782227
4: -100.9121475, 90.8883057, -184.5755615, 140.5543518, -241.4664917, 275.4638672

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3720011, upper bound: 187.3663691
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3722293, upper bound: 187.3663691
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -84.4544296, 90.4469299, -90.0296860, 95.7459259, -180.2003479, 180.4766235
1: -66.2400513, 85.6331940, -70.5223999, 90.7102203, -156.9502716, 156.1555939
2: -96.1552505, 94.9304047, -102.4902802, 100.3906708, -196.5458679, 197.4206848
3: -45.9472961, 102.2089539, -48.7527466, 108.2179565, -154.1652527, 150.9616699
4: -107.8291397, 93.7583084, -114.9058533, 99.2772675, -207.1064148, 208.6641541

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3713006, upper bound: 187.5046185
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3713006, upper bound: 187.5046188
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -84.4544296, 90.4469299, -146.0465698, 134.7337646, -219.1882019, 236.4934998
1: -66.2400513, 85.6331940, -114.2232513, 127.2852173, -193.5252533, 199.8564453
2: -96.1552505, 94.9304047, -165.4884644, 141.3374176, -237.4925995, 260.4188843
3: -45.9472961, 102.2089539, -69.3965378, 166.8359833, -212.7832642, 171.6054993
4: -107.8291397, 93.7583084, -184.5755615, 140.5543518, -248.3834229, 278.3338318

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3715287, upper bound: 187.5068508
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3715288, upper bound: 187.5046188
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -77.2794342, 82.7829361, -172.2308655, 167.1613464
1: -69.8412399, 84.1715393, -60.2073326, 77.6080170, -147.4492340, 144.3788452
2: -101.2739258, 94.6007919, -87.3915558, 87.7073898, -188.9813080, 181.9923401
3: -45.8811989, 105.0105896, -42.6639175, 91.9010391, -137.7822113, 147.6744995
4: -113.0568390, 94.2403870, -97.7812576, 86.9560318, -200.0128326, 192.0216370

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.2381998, upper bound: 187.4330559
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.2381998, upper bound: 187.7207162
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -79.5578537, 84.0953522, -173.5433350, 169.4397583
1: -69.8412399, 84.1715393, -62.0749130, 78.7678299, -148.6090393, 146.2463989
2: -101.2739258, 94.6007919, -90.0511703, 89.0322723, -190.3061981, 184.6519623
3: -45.8811989, 105.0105896, -43.3339539, 94.1345062, -140.0157013, 148.3445435
4: -113.0568390, 94.2403870, -100.6625519, 88.4129562, -201.4697418, 194.9029388

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4378555, upper bound: 187.4378555
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4378555, upper bound: 187.7255157
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -40.6606750, 54.5537338, -144.0016632, 130.5425720
1: -69.8412399, 84.1715393, -31.8599815, 51.1444206, -120.9856415, 116.0315170
2: -101.2739258, 94.6007919, -46.6601028, 57.5620499, -158.8359680, 141.2608948
3: -45.8811989, 105.0105896, -27.1700439, 54.6472054, -100.5284042, 132.1806335
4: -113.0568390, 94.2403870, -52.6399651, 56.9226837, -169.9795227, 146.8803558

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3895086, upper bound: 187.8131253
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8143927, upper bound: 187.8120716
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -88.8516617, 89.4957199, -178.9436951, 178.7335663
1: -69.8412399, 84.1715393, -69.3641663, 83.8208542, -153.6620636, 153.5356750
2: -101.2739258, 94.6007919, -100.5895386, 94.2378540, -195.5117188, 195.1903381
3: -45.8811989, 105.0105896, -45.7091484, 104.3674927, -150.2486725, 150.7197418
4: -113.0568390, 94.2403870, -112.3173523, 93.8545685, -206.9114075, 206.5577393

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8184764, upper bound: 187.7315467
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7729422, upper bound: 187.8030113
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8123064, upper bound: 187.8099099
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -143.1016235, 130.5498047, -78.3078232, 83.3920822, -226.4936829, 208.8576355
1: -112.0695267, 123.2332306, -61.0082550, 78.1903000, -190.2598267, 184.2414856
2: -162.2405090, 136.9816742, -88.5467072, 88.3264847, -250.5669708, 225.5283508
3: -67.0209732, 163.5315094, -42.9643326, 92.9597931, -159.9807739, 206.4958344
4: -180.7981262, 136.1227417, -99.0556564, 87.5924988, -268.3906250, 235.1784058

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3441154, upper bound: 187.6223375
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4397656, upper bound: 187.6198411
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -143.1016235, 130.5498047, -93.0833359, 92.5407257, -235.6423492, 223.6331482
1: -112.0695267, 123.2332306, -72.7047272, 86.7016907, -198.7712097, 195.9379272
2: -162.2405090, 136.9816742, -105.3617554, 97.4237671, -259.6642761, 242.3433838
3: -67.0209732, 163.5315094, -47.1332130, 109.0123520, -176.0333252, 210.6647186
4: -180.7981262, 136.1227417, -117.6394958, 97.0536499, -277.8517761, 253.7622375

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6809643, upper bound: 187.6265696
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7312956, upper bound: 187.6112392
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -144.1855774, 131.0563660, -93.0833359, 92.5407257, -236.7263031, 224.1397095
1: -112.9922714, 123.7403946, -72.7047272, 86.7016907, -199.6939697, 196.4450684
2: -163.5684509, 137.5474701, -105.3617554, 97.4237671, -260.9922180, 242.9092255
3: -67.3457336, 164.4682617, -47.1332130, 109.0123520, -176.3580933, 211.6014709
4: -182.2239990, 136.7161255, -117.6394958, 97.0536499, -279.2776489, 254.3556213

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4691121, upper bound: 187.5379804
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7203645, upper bound: 187.5370434
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -87.2105865, 93.5698242, -183.0177612, 177.0924835
1: -69.8412399, 84.1715393, -68.2377090, 88.7802505, -158.6214600, 152.4092407
2: -101.2739258, 94.6007919, -99.1996841, 98.2294769, -199.5033722, 193.8004761
3: -45.8811989, 105.0105896, -47.5788269, 105.0312576, -150.9124298, 152.4107819
4: -113.0568390, 94.2403870, -111.2859421, 96.9314651, -209.9883118, 205.5263367

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6059226, upper bound: 187.7646796
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6059226, upper bound: 187.8071747
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -89.0158539, 95.0906677, -184.5386200, 178.8977356
1: -69.8412399, 84.1715393, -69.7114868, 90.0992203, -159.9404602, 153.8830109
2: -101.2739258, 94.6007919, -101.3397827, 99.7479248, -201.0218506, 195.9405823
3: -45.8811989, 105.0105896, -48.4435043, 107.1256180, -153.0068207, 153.3172607
4: -113.0568390, 94.2403870, -113.6350861, 98.5864563, -211.6432953, 207.8754730

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3743268, upper bound: 187.8085892
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262965, upper bound: 187.8075679
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -67.0772324, 75.0328751, -146.0465698, 134.7337646, -201.8110046, 221.0794373
1: -52.3760071, 70.2153320, -114.2232513, 127.2852173, -179.6612244, 184.4385834
2: -76.1112671, 79.5657120, -165.4884644, 141.3374176, -217.4486237, 245.0541687
3: -38.7699432, 81.0885849, -69.3965378, 166.8359833, -205.6058960, 150.4851074
4: -85.1397247, 78.6775513, -184.5755615, 140.5543518, -225.6940613, 263.2531128

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289116, upper bound: 187.4892030
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277554, upper bound: 187.6841739
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -87.1010132, 88.2613983, -146.0465698, 134.7337646, -221.8347778, 234.3079681
1: -67.9715729, 82.6533890, -114.2232513, 127.2852173, -195.2567902, 196.8766479
2: -98.5855026, 92.9570694, -165.4884644, 141.3374176, -239.9228668, 258.4455261
3: -45.1060333, 102.4525528, -69.3965378, 166.8359833, -211.9420166, 171.8490906
4: -110.1000519, 92.5308685, -184.5755615, 140.5543518, -250.6544037, 277.1064453

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294540, upper bound: 187.4892030
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281608, upper bound: 187.8148329
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -80.3507690, 90.7024994, -234.9692841, 211.5807495
1: -112.9824219, 123.8839874, -63.1035271, 85.9116058, -198.8940277, 186.9875031
2: -163.5588074, 137.6753693, -91.7921677, 95.2717209, -258.8305359, 229.4675140
3: -67.3619995, 164.7313690, -46.2828140, 98.7244492, -166.0864258, 210.7783203
4: -182.2491913, 136.8414307, -103.1749496, 93.9415741, -276.1907654, 240.0163422

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3743426, upper bound: 187.4658970
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3743425, upper bound: 187.6259132
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -87.0424194, 93.9333038, -238.2000580, 218.2723694
1: -112.9824219, 123.8839874, -68.1751099, 88.9729767, -201.9553986, 192.0590973
2: -163.5588074, 137.6753693, -99.1208801, 98.4945221, -262.0533142, 236.7962341
3: -67.3619995, 164.7313690, -47.8467789, 105.1401672, -172.5021362, 212.4054871
4: -182.2491913, 136.8414307, -111.2113953, 97.3413239, -279.5904846, 248.0528107

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6064999, upper bound: 187.6208874
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263098, upper bound: 187.6259132
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -133.9066010, 125.3042831, -146.0465698, 134.7337646, -268.6403809, 271.3507690
1: -105.0086517, 118.3775406, -114.2232513, 127.2852173, -232.2938690, 232.6007996
2: -151.9803162, 131.6812744, -165.4884644, 141.3374176, -293.3177490, 297.1697388
3: -64.3408432, 154.3021545, -69.3965378, 166.8359833, -231.1767883, 223.6987000
4: -169.5246887, 130.5524292, -184.5755615, 140.5543518, -310.0790405, 315.1279907

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6287907, upper bound: 187.4539135
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291594, upper bound: 187.4670389
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294333, upper bound: 187.4588856
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -140.4060669, 128.7797394, -146.0465698, 134.7337646, -275.1398315, 274.8262939
1: -109.9443588, 121.5496521, -114.2232513, 127.2852173, -237.2295685, 235.7728882
2: -159.1761932, 135.1240082, -165.4884644, 141.3374176, -300.5136108, 300.6124878
3: -66.1132355, 160.6915894, -69.3965378, 166.8359833, -232.9491882, 230.0881348
4: -177.4199219, 134.2343445, -184.5755615, 140.5543518, -317.9742737, 318.8098145

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277421, upper bound: 187.5403019
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4352229, upper bound: 187.6264776
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4352229, upper bound: 187.6264776
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -93.4733276, 92.8134308, -79.2864609, 88.0104218, -181.4837341, 172.0998688
1: -73.0208511, 86.9460297, -62.3454056, 83.3233795, -156.3442230, 149.2914124
2: -105.8147049, 97.6726913, -90.5679855, 92.4689178, -198.2835999, 188.2406769
3: -47.2495461, 109.4501266, -44.6927567, 97.4578094, -144.7073212, 154.1428680
4: -118.1232986, 97.3208160, -101.7387695, 91.1533737, -209.2766418, 199.0595245

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3375439, upper bound: 187.7848111
time: 0.58 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3640221, upper bound: 187.8149339
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3640221, upper bound: 187.8149339
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -93.4733276, 92.8134308, -85.4159851, 90.9214935, -184.3948212, 178.2293854
1: -73.0208511, 86.9460297, -67.0133209, 86.0685425, -159.0893860, 153.9593506
2: -105.8147049, 97.6726913, -97.3060455, 95.3715820, -201.1862793, 194.9787292
3: -47.2495461, 109.4501266, -46.1031952, 103.3089905, -150.5585327, 155.5533142
4: -118.1232986, 97.3208160, -109.0886841, 94.2118530, -212.3351440, 206.4094849

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4780256, upper bound: 187.7840620
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5068518, upper bound: 187.4873317
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068518, upper bound: 187.8142334
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -90.7469177, 90.6126022, -141.1933289, 135.6199799, -225.2737732, 231.8059387
1: -70.7987976, 84.9603500, -110.6310196, 128.9560394, -197.8499298, 195.5913696
2: -102.6348648, 95.4938202, -160.5225830, 141.7883911, -242.2296753, 256.0163879
3: -46.0043831, 106.2343369, -69.0409546, 163.8030243, -209.8074036, 173.2877350
4: -114.5984802, 95.0162354, -179.4125366, 140.3392334, -253.7844849, 274.4287720

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5998729, upper bound: 187.7649222
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5998729, upper bound: 187.7649222
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -90.7469177, 90.6126022, -142.7752991, 136.8418579, -226.7329407, 233.3878784
1: -70.7987976, 84.9603500, -111.9555359, 130.0005951, -199.0739594, 196.9158936
2: -102.6348648, 95.4938202, -162.4290619, 142.9362183, -243.5780334, 257.9228821
3: -46.0043831, 106.2343369, -69.7176361, 165.6833496, -211.6877289, 174.0357513
4: -114.5984802, 95.0162354, -181.5003052, 141.6623383, -255.3515167, 276.5164795

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5998729, upper bound: 187.7740562
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5998729, upper bound: 187.7740562
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -81.7393875, 85.8541565, -143.9209900, 137.5663147, -218.1754608, 229.7751160
1: -63.9643745, 80.5138931, -112.8730698, 130.6749878, -192.7401123, 193.3869629
2: -92.7056122, 90.6816025, -163.7390137, 143.6699066, -234.1477051, 254.4206238
3: -43.8189468, 97.5556946, -70.0665512, 166.9166107, -210.7355652, 165.5838165
4: -103.7032394, 89.9974823, -182.9464722, 142.4316864, -244.9097900, 272.9439697

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6062632, upper bound: 187.4807999
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6062632, upper bound: 187.4879400
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -89.1911621, 90.0023575, -143.9209900, 137.5663147, -225.8434448, 233.9233093
1: -69.6335144, 84.3171387, -112.8730698, 130.6749878, -198.5485840, 197.1902161
2: -100.9520874, 94.8109741, -163.7390137, 143.6699066, -242.5951996, 258.5499878
3: -45.8655128, 104.9505463, -70.0665512, 166.9166107, -212.7821198, 173.0806274
4: -112.7668915, 94.3528519, -182.9464722, 142.4316864, -254.2122192, 277.2993164

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6051175, upper bound: 187.8027840
time: 0.54 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6051175, upper bound: 187.8076470
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -79.2864609, 88.0104218, -232.2771912, 210.5164490
1: -112.9824219, 123.8839874, -62.3454056, 83.3233795, -196.3058014, 186.2293854
2: -163.5588074, 137.6753693, -90.5679855, 92.4689178, -256.0277100, 228.2433319
3: -67.3619995, 164.7313690, -44.6927567, 97.4578094, -164.8198090, 209.4241180
4: -182.2491913, 136.8414307, -101.7387695, 91.1533737, -273.4025269, 238.5801697

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3635645, upper bound: 187.4659113
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3635645, upper bound: 187.6259274
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -85.4159851, 90.9214935, -235.1882782, 216.6459656
1: -112.9824219, 123.8839874, -67.0133209, 86.0685425, -199.0509644, 190.8973083
2: -163.5588074, 137.6753693, -97.3060455, 95.3715820, -258.9303894, 234.9813995
3: -67.3619995, 164.7313690, -46.1031952, 103.3089905, -170.6709900, 210.8345642
4: -182.2491913, 136.8414307, -109.0886841, 94.2118530, -276.4610291, 245.9300995

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3635645, upper bound: 187.4652108
time: 0.58 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3635645, upper bound: 187.6252269
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -141.1933289, 135.6199799, -278.9478149, 272.4233093
1: -112.9824219, 123.8839874, -110.6310196, 128.9560394, -240.1585388, 234.5149841
2: -163.5588074, 137.6753693, -160.5225830, 141.7883911, -303.2407532, 298.1979065
3: -67.3619995, 164.7313690, -69.0409546, 163.8030243, -231.1650238, 231.7338409
4: -182.2491913, 136.8414307, -179.4125366, 140.3392334, -321.6118774, 316.2539368

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6043117, upper bound: 187.5554372
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6043118, upper bound: 187.6218485
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -144.2667847, 131.2299805, -142.7752991, 136.8418579, -280.4069519, 274.0052795
1: -112.9824219, 123.8839874, -111.9555359, 130.0005951, -241.3825989, 235.8395233
2: -163.5588074, 137.6753693, -162.4290619, 142.9362183, -304.5890808, 300.1044312
3: -67.3619995, 164.7313690, -69.7176361, 165.6833496, -233.0453491, 232.4818573
4: -182.2491913, 136.8414307, -181.5003052, 141.6623383, -323.1789551, 318.3417053

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240268, upper bound: 187.5614963
time: 0.49 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240268, upper bound: 187.6279076
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -78.3078232, 83.3920822, -143.5740662, 130.9119568, -209.2197876, 226.9661255
1: -61.0082550, 78.1903000, -112.4834366, 123.5442581, -184.5525208, 190.6737366
2: -88.5467072, 88.3264847, -162.7946777, 137.3177643, -225.8644409, 251.1211243
3: -42.9643326, 92.9597931, -67.2107239, 164.0457764, -207.0101013, 160.1705170
4: -99.0556564, 87.5924988, -181.3767090, 136.5292358, -235.5848999, 268.9692078

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6223375, upper bound: 187.3441154
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6198411, upper bound: 187.4397656
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -93.0833359, 92.5407257, -143.5740662, 130.9119568, -223.9953003, 236.1147919
1: -72.7047272, 86.7016907, -112.4834366, 123.5442581, -196.2489777, 199.1851196
2: -105.3617554, 97.4237671, -162.7946777, 137.3177643, -242.6794739, 260.2184143
3: -47.1332130, 109.0123520, -67.2107239, 164.0457764, -211.1789856, 176.2230835
4: -117.6394958, 97.0536499, -181.3767090, 136.5292358, -254.1687012, 278.4303589

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265696, upper bound: 187.6809643
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6112392, upper bound: 187.7312956
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -93.0833359, 92.5407257, -145.2095184, 131.8752594, -224.9585876, 237.7502441
1: -72.7047272, 86.7016907, -113.8639603, 124.4382782, -197.1430054, 200.5656433
2: -105.3617554, 97.4237671, -164.7864685, 138.2937317, -243.6554718, 262.2102356
3: -47.1332130, 109.0123520, -67.7423248, 165.6072845, -212.7404938, 176.7546692
4: -117.6394958, 97.0536499, -183.4982605, 137.6233063, -255.2628021, 280.5519104

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5379804, upper bound: 187.4691121
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5370434, upper bound: 187.7203645
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -78.3078232, 83.3920822, -209.0372162, 182.2736664, -260.5814819, 292.4292908
1: -61.0082550, 78.1903000, -164.1908569, 172.4178772, -233.4261322, 242.3811646
2: -88.5467072, 88.3264847, -237.1544189, 190.1957245, -278.7424316, 325.4808960
3: -42.9643326, 92.9597931, -93.8365173, 234.9767609, -277.9411011, 185.8746643
4: -99.0556564, 87.5924988, -263.9682312, 190.0406952, -289.0963440, 351.5606995

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6229894, upper bound: 187.4387862
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6230828, upper bound: 187.3490219
time: 0.53 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6205864, upper bound: 187.4446721
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.0833359, 92.5407257, -209.0372162, 182.2736664, -275.3569946, 301.5778809
1: -72.7047272, 86.7016907, -164.1908569, 172.4178772, -245.1226044, 250.8925476
2: -105.3617554, 97.4237671, -237.1544189, 190.1957245, -295.5574951, 334.5781860
3: -47.1332130, 109.0123520, -93.8365173, 234.9767609, -282.1099548, 202.0666504
4: -117.6394958, 97.0536499, -263.9682312, 190.0406952, -307.6801758, 361.0218811

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6273345, upper bound: 187.6890196
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6120040, upper bound: 187.7393509
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.0833359, 92.5407257, -209.4814911, 182.5418701, -275.6252136, 302.0222168
1: -72.7047272, 86.7016907, -164.6514130, 172.6817017, -245.3864136, 251.3530731
2: -105.3617554, 97.4237671, -237.7784882, 190.4826355, -295.8443909, 335.2021790
3: -47.1332130, 109.0123520, -94.0311890, 235.3506622, -282.4838562, 202.2289886
4: -117.6394958, 97.0536499, -264.5895691, 190.3989410, -308.0384521, 361.6432190

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5411571, upper bound: 187.4700001
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5402201, upper bound: 187.7212525
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -140.4060669, 128.7797394, -144.8092651, 131.6590424, -272.0651245, 273.5889893
1: -109.9443588, 121.5496521, -113.4549103, 124.2506790, -234.1950226, 235.0045624
2: -159.1761932, 135.1240082, -164.1941833, 138.0722961, -297.2484741, 299.3181763
3: -66.1132355, 160.6915894, -67.5839310, 165.3264618, -231.4396973, 228.2755127
4: -177.4199219, 134.2343445, -182.9160309, 137.3219604, -314.7418213, 317.1503601

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266290, upper bound: 187.5402970
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5390409, upper bound: 187.5357447
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -121.1534882, 118.1547852, -210.2807465, 183.0075378, -304.1610107, 328.4355469
1: -94.9650574, 111.6696167, -165.1631775, 173.1105804, -268.0756226, 276.8327026
2: -137.2637482, 124.6230392, -238.5625763, 190.9392853, -328.2030334, 363.1856079
3: -61.0121613, 140.4321442, -94.2064362, 236.2539673, -297.2661133, 233.8416443
4: -153.3799286, 123.1392441, -265.5162659, 190.8201752, -344.2001038, 388.6555176

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239300, upper bound: 187.3311088
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5334987, upper bound: 187.3286602
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -127.2982178, 121.3441238, -210.2807465, 183.0075378, -310.3057556, 331.6248779
1: -99.5095673, 114.5703125, -165.1631775, 173.1105804, -272.6201477, 279.7334595
2: -144.1310272, 127.7496490, -238.5625763, 190.9392853, -335.0703125, 366.3121948
3: -62.6166992, 146.5494232, -94.2064362, 236.2539673, -298.8706665, 240.0565338
4: -160.8834839, 126.4956894, -265.5162659, 190.8201752, -351.7036438, 392.0119629

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6212358, upper bound: 187.4186829
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5305338, upper bound: 187.4162473
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -88.1659546, 92.5673523, -210.2807465, 183.0075378, -271.1734924, 302.8480835
1: -69.1605377, 87.6359100, -165.1631775, 173.1105804, -242.2711182, 252.7990723
2: -100.3883057, 97.0811081, -238.5625763, 190.9392853, -291.3275757, 335.6436768
3: -46.9165192, 106.1280746, -94.2064362, 236.2539673, -283.1704712, 199.6428375
4: -112.4864349, 95.9681168, -265.5162659, 190.8201752, -303.3066101, 361.4843750

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3652667
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5068309
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -143.6635437, 130.8681641, -210.2807465, 183.0075378, -326.6710510, 341.1489258
1: -112.4993591, 123.5508575, -165.1631775, 173.1105804, -285.6099243, 288.7139893
2: -162.8639221, 137.3267822, -238.5625763, 190.9392853, -353.8032227, 375.8893433
3: -67.1903763, 164.0828094, -94.2064362, 236.2539673, -303.4443359, 257.4977722
4: -181.4958038, 136.4762421, -265.5162659, 190.8201752, -372.3159790, 401.9924927

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239400, upper bound: 187.4657321
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232395, upper bound: 187.6264775
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -67.2816544, 82.2667694, -93.5899734, 92.8750916, -160.1567383, 175.8567505
1: -52.9685898, 77.8987427, -73.1107025, 87.0040359, -139.9726257, 151.0094452
2: -77.1924667, 86.4871979, -105.9443512, 97.7353897, -174.9278564, 192.4315338
3: -41.8829956, 85.2084885, -47.2811356, 109.5639648, -151.1113586, 132.4896240
4: -87.0138321, 85.0197067, -118.2672424, 97.3865814, -184.4004059, 203.2869415

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7842324, upper bound: 186.8322268
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8136326, upper bound: 186.8628891
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8136326, upper bound: 186.8652443
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -72.8509216, 85.0178375, -93.5899734, 92.8750916, -165.7260132, 178.6078033
1: -57.2537613, 80.4937592, -73.1107025, 87.0040359, -144.2577667, 153.6044617
2: -83.4400635, 89.2036133, -105.9443512, 97.7353897, -181.1754456, 195.1479645
3: -43.2070618, 90.6576691, -47.2811356, 109.5639648, -152.4847717, 137.9388123
4: -93.7365036, 87.8741302, -118.2672424, 97.3865814, -191.1230621, 206.1413422

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7841297, upper bound: 187.4015864
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8012698, upper bound: 187.4303827
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8062185, upper bound: 187.4010820
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -80.2037811, 90.6093750, -93.5899734, 92.8750916, -173.0788574, 184.1993408
1: -62.9866447, 85.8226929, -73.1107025, 87.0040359, -149.9906769, 158.9333954
2: -91.6254730, 95.1752701, -105.9443512, 97.7353897, -189.3608704, 201.1195984
3: -46.2366257, 98.5653610, -47.2811356, 109.5639648, -155.5843964, 145.8464966
4: -102.9899521, 93.8429642, -118.2672424, 97.3865814, -200.3765259, 212.1101837

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7865031, upper bound: 187.3455306
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8161505, upper bound: 187.3719788
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8159557, upper bound: 187.3743339
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -86.8900909, 93.8365326, -93.5899734, 92.8750916, -179.7651825, 187.4264832
1: -68.0529099, 88.8813324, -73.1107025, 87.0040359, -155.0569458, 161.9920044
2: -98.9465790, 98.3943405, -105.9443512, 97.7353897, -196.6819763, 204.3386841
3: -47.7982101, 104.9755630, -47.2811356, 109.5639648, -157.2085876, 152.2566986
4: -111.0191040, 97.2381668, -118.2672424, 97.3865814, -208.4056854, 215.5053558

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8023840, upper bound: 187.6064866
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8075197, upper bound: 187.6262966
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -116.9704285, 121.8325043, -93.5899734, 92.8750916, -209.8455200, 212.8580170
1: -91.8533173, 115.7528610, -73.1107025, 87.0040359, -178.8573456, 185.7792816
2: -133.2842560, 127.3907318, -105.9443512, 97.7353897, -231.0196533, 229.6474457
3: -62.0518303, 138.7779999, -47.2811356, 109.5639648, -168.9657288, 186.0591431
4: -149.5080414, 125.8340988, -118.2672424, 97.3865814, -246.8946075, 241.3683472

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3170833, upper bound: 186.8505771
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8159108, upper bound: 186.8935726
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -123.2414398, 125.1088715, -93.5899734, 92.8750916, -216.1165161, 216.2842865
1: -96.7073822, 118.8619843, -73.1107025, 87.0040359, -183.7114105, 189.0157776
2: -140.4173279, 130.6635284, -105.9443512, 97.7353897, -238.1527100, 233.0637512
3: -63.6406441, 145.4647217, -47.2811356, 109.5639648, -170.6586456, 192.7458496
4: -157.1356964, 129.2418365, -118.2672424, 97.3865814, -254.5222778, 244.9175720

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3163305, upper bound: 187.3765693
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8151581, upper bound: 187.4195648
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -143.7625427, 137.4667053, -90.7469177, 90.6126022, -234.3751221, 227.3360748
1: -112.7459106, 130.5810547, -70.7987976, 84.9603500, -197.7062683, 199.6336365
2: -163.5565948, 143.5669098, -102.6348648, 95.4938202, -259.0504150, 244.1869202
3: -70.0163727, 166.7445679, -46.0043831, 106.2343369, -174.3347778, 212.7489471
4: -182.7444916, 142.3251953, -114.5984802, 95.0162354, -277.7606812, 255.9902496

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7647832, upper bound: 187.6045601
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7647832, upper bound: 187.6242752
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -143.7625427, 137.4667053, -92.5626678, 92.1819839, -235.9445190, 229.1054535
1: -112.7459106, 130.5810547, -72.2883682, 86.3782349, -199.1241455, 201.0934601
2: -163.5565948, 143.5669098, -104.7752304, 97.0700226, -260.6266174, 246.3023071
3: -70.0163727, 166.7445679, -46.9766197, 108.4464035, -176.5135651, 213.7211914
4: -182.7444916, 142.3251953, -116.9716492, 96.6647339, -279.4092407, 258.3150330

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4879705, upper bound: 187.6271832
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8099546, upper bound: 187.6260375
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -67.2816544, 82.2667694, -149.9745941, 137.3406982, -204.6223297, 232.2413635
1: -52.9685898, 77.8987427, -117.3068619, 129.7829132, -182.7514954, 195.2055969
2: -77.1924667, 86.4871979, -169.9014282, 144.0322723, -221.2247314, 256.3886108
3: -41.8829956, 85.2084885, -70.6789017, 171.0912170, -212.7890167, 155.8873901
4: -87.0138321, 85.0197067, -189.4996490, 143.3141022, -230.3279266, 274.5193481

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6229228, upper bound: 186.8630066
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6229228, upper bound: 186.8652439
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -72.8509216, 85.0178375, -149.9745941, 137.3406982, -210.1916199, 234.9924316
1: -57.2537613, 80.4937592, -117.3068619, 129.7829132, -187.0366516, 197.8006134
2: -83.4400635, 89.2036133, -169.9014282, 144.0322723, -227.4722900, 259.1050415
3: -43.2070618, 90.6576691, -70.6789017, 171.0912170, -214.1624298, 161.3365784
4: -93.7365036, 87.8741302, -189.4996490, 143.3141022, -237.0505981, 277.3737488

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6207281, upper bound: 187.4303827
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256769, upper bound: 187.4010820
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -80.2037811, 90.6093750, -149.9745941, 137.3406982, -217.5444489, 240.5839691
1: -62.9866447, 85.8226929, -117.3068619, 129.7829132, -192.7695312, 203.1295471
2: -91.6254730, 95.1752701, -169.9014282, 144.0322723, -235.6576843, 265.0766907
3: -46.2366257, 98.5653610, -70.6789017, 171.0912170, -217.2620544, 169.2442627
4: -102.9899521, 93.8429642, -189.4996490, 143.3141022, -246.3040466, 283.3426208

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251722, upper bound: 187.3720391
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251722, upper bound: 187.3743333
time: 0.55 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.25 + 418.46 = 420.71 seconds
