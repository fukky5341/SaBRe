## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.77104719722996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423)
1: (-124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621)
2: (-105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148)
3: (-110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960)
4: (-94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.71 + 2.26 = 2.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8050277, upper bound: 339.8050277

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8049169, upper bound: 339.8049626
time: 1.17 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8050253, upper bound: 339.8050253
time: 0.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.14 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 0, lower bound: -339.8049169, upper bound: 339.8049626
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 0, lower bound: -339.8050253, upper bound: 339.8050253

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -88.6961975, 297.1046753, -87.3910828, 292.2948608, -380.9910583, 384.4957581
1: -124.4471970, 294.8176575, -122.5927658, 290.1781006, -414.6253052, 417.4103699
2: -105.5478058, 324.6724243, -103.9885712, 319.6148682, -425.1626587, 428.6610107
3: -110.7164154, 421.9519958, -109.0454254, 415.1443787, -525.8607788, 530.9974365
4: -94.5076294, 383.5692749, -93.0869598, 377.4335022, -471.9411316, 476.6562500

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8045398, upper bound: 339.8040898
time: 1.02 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 0.89 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -88.6961975, 297.1046753, -88.3216934, 295.8016357, -384.4978027, 385.4263611
1: -124.4471970, 294.8176575, -123.9163055, 293.5386963, -417.9859009, 418.7339478
2: -105.5478058, 324.6724243, -105.1011658, 323.2695618, -428.8173828, 429.7735596
3: -110.7164154, 421.9519958, -110.2443161, 420.1114197, -530.8278198, 532.1962891
4: -94.5076294, 383.5692749, -94.1079941, 381.9024048, -476.4100342, 477.6772766

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040898, upper bound: 339.8046282
time: 0.94 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.42 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -339.8045398, upper bound: 339.8040898
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -339.8040898, upper bound: 339.8046282
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -87.5877380, 293.3702698, -83.9690399, 279.8752136, -367.4629517, 377.3392944
1: -122.9078217, 291.1408081, -117.7797165, 277.9726257, -400.8804321, 408.9205017
2: -104.2449875, 320.6318665, -99.9207306, 306.2253113, -410.4703064, 420.5525818
3: -109.3434143, 416.6802368, -104.7510605, 397.5820312, -506.9254456, 521.4312744
4: -93.3429947, 378.7810364, -89.4517059, 361.6362610, -454.9792480, 468.2327271

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 1.17 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 1.00 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -87.4125977, 293.1314087, -84.1043701, 282.1285706, -369.5411682, 377.2357788
1: -122.7049026, 290.8285217, -118.1360779, 279.9616394, -402.6665344, 408.9645996
2: -104.0567322, 320.2686768, -100.1703873, 308.3364563, -412.3931885, 420.4390564
3: -109.1592331, 416.3205872, -105.0615692, 400.7293396, -509.8885803, 521.3821411
4: -93.1842804, 378.4085083, -89.6990509, 364.2296143, -457.4138794, 468.1075439

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 0.75 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 1.05 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -87.2101364, 292.0566406, -377.1516113, 371.3806152
1: -119.3920059, 282.0884705, -122.3726425, 289.8511047, -409.2430725, 404.4610901
2: -101.2751236, 310.7092590, -103.7945404, 319.2184143, -420.4934998, 414.5037842
3: -106.2089005, 403.8296509, -108.8673325, 414.8243713, -521.0332642, 512.6969604
4: -90.6926956, 367.2429504, -92.9399033, 377.1012573, -467.7939453, 460.1828613

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 0.92 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 0.87 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -87.0423737, 291.8428650, -377.3496399, 374.2844543
1: -120.1114273, 284.9082947, -122.1802826, 289.5643311, -409.6757202, 407.0885620
2: -101.8361511, 313.7402954, -103.6151733, 318.8814697, -420.7176208, 417.3554382
3: -106.8467102, 407.9730225, -108.6925430, 414.5008240, -521.3474731, 516.6655884
4: -91.2148666, 370.7578735, -92.7891159, 376.7607117, -467.9755859, 463.5469971

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 1.01 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 1.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.82 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -83.9690399, 279.8752136, -364.9701843, 368.1395264
1: -119.3920059, 282.0884705, -117.7797165, 277.9726257, -397.3646240, 399.8681946
2: -101.2751236, 310.7092590, -99.9207306, 306.2253113, -407.5003967, 410.6300049
3: -106.2089005, 403.8296509, -104.7510605, 397.5820312, -503.7909241, 508.5807190
4: -90.6926956, 367.2429504, -89.4517059, 361.6362610, -452.3289490, 456.6946411

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973503, upper bound: 339.8010500
time: 1.22 seconds

## Relational analysis of NS_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8045398, upper bound: 339.8040898
time: 0.96 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -83.9690399, 279.8752136, -365.3819885, 371.2111206
1: -120.1114273, 284.9082947, -117.7797165, 277.9726257, -398.0840454, 402.6880188
2: -101.8361511, 313.7402954, -99.9207306, 306.2253113, -408.0614624, 413.6610107
3: -106.8467102, 407.9730225, -104.7510605, 397.5820312, -504.4287415, 512.7240601
4: -91.2148666, 370.7578735, -89.4517059, 361.6362610, -452.8511353, 460.2095642

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973503, upper bound: 339.8010500
time: 1.57 seconds

## Relational analysis of NS_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8045398, upper bound: 339.8040898
time: 1.05 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -84.1043701, 282.1285706, -367.2235413, 368.2748413
1: -119.3920059, 282.0884705, -118.1360779, 279.9616394, -399.3536377, 400.2245178
2: -101.2751236, 310.7092590, -100.1703873, 308.3364563, -409.6115723, 410.8796387
3: -106.2089005, 403.8296509, -105.0615692, 400.7293396, -506.9382324, 508.8912354
4: -90.6926956, 367.2429504, -89.6990509, 364.2296143, -454.9223022, 456.9419861

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021021, upper bound: 339.7967364
time: 2.00 seconds

## Relational analysis of NS_B1_B2_A1_B2

### Relational analysis result of NS_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 1.37 seconds

## BFS NS instance: NS_B1_B2_A2

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -84.1043701, 282.1285706, -367.6353455, 371.3464661
1: -120.1114273, 284.9082947, -118.1360779, 279.9616394, -400.0730591, 403.0443726
2: -101.8361511, 313.7402954, -100.1703873, 308.3364563, -410.1726074, 413.9106750
3: -106.8467102, 407.9730225, -105.0615692, 400.7293396, -507.5760498, 513.0346069
4: -91.2148666, 370.7578735, -89.6990509, 364.2296143, -455.4444885, 460.4568787

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A2_A1

### Relational analysis result of NS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7968124, upper bound: 339.8005007
time: 0.86 seconds

## Relational analysis of NS_B1_B2_A2_A2

### Relational analysis result of NS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
time: 1.09 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -84.7256393, 282.8713074, -367.9662781, 368.8961182
1: -119.3920059, 282.0884705, -118.8686752, 280.8153687, -400.2073364, 400.9571533
2: -101.2751236, 310.7092590, -100.8353577, 309.3188782, -410.5939636, 411.5446167
3: -106.2089005, 403.8296509, -105.7429886, 401.9952698, -508.2041626, 509.5726318
4: -90.6926956, 367.2429504, -90.2985229, 365.5823364, -456.2750244, 457.5414429

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023517, upper bound: 339.7976404
time: 1.13 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040898, upper bound: 339.8046282
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -85.1356964, 285.9497681, -371.0447083, 369.3061523
1: -119.3920059, 282.0884705, -119.5853043, 283.6409607, -403.0329590, 401.6737366
2: -101.2751236, 310.7092590, -101.3927994, 312.3492126, -413.6242676, 412.1020508
3: -106.2089005, 403.8296509, -106.3785629, 406.1484680, -512.3573608, 510.2081909
4: -90.6926956, 367.2429504, -90.8181458, 369.1052551, -459.7979431, 458.0610962

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023517, upper bound: 339.7976404
time: 0.94 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040898, upper bound: 339.8046282
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -84.7256393, 282.8713074, -368.3780823, 371.9677429
1: -120.1114273, 284.9082947, -118.8686752, 280.8153687, -400.9267883, 403.7769775
2: -101.8361511, 313.7402954, -100.8353577, 309.3188782, -411.1550293, 414.5756531
3: -106.8467102, 407.9730225, -105.7429886, 401.9952698, -508.8419800, 513.7160034
4: -91.2148666, 370.7578735, -90.2985229, 365.5823364, -456.7972107, 461.0563660

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7969932, upper bound: 339.8017825
time: 1.04 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 1.48 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -85.1356964, 285.9497681, -371.4565430, 372.3777771
1: -120.1114273, 284.9082947, -119.5853043, 283.6409607, -403.7523499, 404.4935608
2: -101.8361511, 313.7402954, -101.3927994, 312.3492126, -414.1853333, 415.1330872
3: -106.8467102, 407.9730225, -106.3785629, 406.1484680, -512.9951782, 514.3515625
4: -91.2148666, 370.7578735, -90.8181458, 369.1052551, -460.3201294, 461.5760193

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017825, upper bound: 339.7969932
time: 1.03 seconds

## Relational analysis of NS_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.50 seconds
NS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.7973503, upper bound: 339.8010500
NS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8045398, upper bound: 339.8040898
NS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.7973503, upper bound: 339.8010500
NS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8045398, upper bound: 339.8040898
NS_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8021021, upper bound: 339.7967364
NS_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.7968124, upper bound: 339.8005007
NS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8039484, upper bound: 339.8040535
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8023517, upper bound: 339.7976404
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8040898, upper bound: 339.8046282
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8023517, upper bound: 339.7976404
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8040898, upper bound: 339.8046282
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.7969932, upper bound: 339.8017825
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162
NS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8017825, upper bound: 339.7969932
NS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -339.8039162, upper bound: 339.8039162

## BFS NS instance: NS_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -83.8959808, 279.6270752, -362.3118896, 359.9349365
1: -115.9796906, 273.9595032, -117.6765442, 277.7276001, -393.7072754, 391.6360474
2: -98.3984375, 301.7886658, -99.8341293, 305.9558411, -404.3542786, 401.6227722
3: -103.1827164, 392.3468323, -104.6593323, 397.2304382, -500.4131470, 497.0061646
4: -88.1445007, 356.8894043, -89.3746719, 361.3166809, -449.4611816, 446.2640686

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8012659
time: 1.18 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8042104
time: 1.97 seconds

## BFS NS instance: NS_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -83.9690399, 279.8752136, -364.1096191, 365.1923828
1: -118.1718445, 279.1607971, -117.7797165, 277.9726257, -396.1444702, 396.9405212
2: -100.2456055, 307.4933777, -99.9207306, 306.2253113, -406.4709167, 407.4141235
3: -105.1194687, 399.6337585, -104.7510605, 397.5820312, -502.7015076, 504.3848267
4: -89.7725525, 363.4640198, -89.4517059, 361.6362610, -451.4088135, 452.9156799

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8048113, upper bound: 339.8012659
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8048113, upper bound: 339.8049209
time: 0.94 seconds

## BFS NS instance: NS_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -83.8959808, 279.6270752, -362.9134216, 363.2489624
1: -116.9414825, 277.0936890, -117.6765442, 277.7276001, -394.6690674, 394.7702332
2: -99.1595917, 305.1754456, -99.8341293, 305.9558411, -405.1153870, 405.0095520
3: -104.0396271, 396.9595032, -104.6593323, 397.2304382, -501.2700806, 501.6188354
4: -88.8424301, 360.8237305, -89.3746719, 361.3166809, -450.1591187, 450.1983948

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A2_A1_A1

### Relational analysis result of NS_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7622247, upper bound: 339.7790232
time: 0.80 seconds

## Relational analysis of NS_B1_B1_A2_A1_A2

### Relational analysis result of NS_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965194, upper bound: 339.8004185
time: 0.65 seconds

## BFS NS instance: NS_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -83.9690399, 279.8752136, -364.5162659, 368.2504272
1: -118.8840942, 281.9696045, -117.7797165, 277.9726257, -396.8567200, 399.7493286
2: -100.8020096, 310.5105286, -99.9207306, 306.2253113, -407.0273132, 410.4312744
3: -105.7517395, 403.7609863, -104.7510605, 397.5820312, -503.3337402, 508.5120544
4: -90.2919159, 366.9395752, -89.4517059, 361.6362610, -451.9281616, 456.3912659

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A2_A2_B1

### Relational analysis result of NS_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8038332, upper bound: 339.8040684
time: 1.05 seconds

## Relational analysis of NS_B1_B1_A2_A2_B2

### Relational analysis result of NS_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973500, upper bound: 339.8029820
time: 0.90 seconds

## BFS NS instance: NS_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -85.0222626, 283.9200745, -81.6958694, 273.5408630, -358.5631104, 365.6159363
1: -119.2890396, 281.8406067, -114.7062836, 271.4855042, -390.7744751, 396.5468750
2: -101.1887512, 310.4375000, -97.2843628, 299.0462952, -400.2350464, 407.7218323
3: -106.1172714, 403.4742737, -102.0274506, 388.6209106, -494.7381897, 505.5017090
4: -90.6156235, 366.9209900, -87.1474457, 353.3419495, -443.9575500, 454.0684204

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B1_A1

### Relational analysis result of NS_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023026, upper bound: 339.7962509
time: 1.05 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2

### Relational analysis result of NS_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8026643, upper bound: 339.7975155
time: 1.29 seconds

## BFS NS instance: NS_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -83.1899185, 278.9436035, -364.0385437, 367.3603821
1: -119.3920059, 282.0884705, -116.8321838, 276.8170471, -396.2089844, 398.9206543
2: -101.2751236, 310.7092590, -99.0710449, 304.8868408, -406.1619263, 409.7802124
3: -106.2089005, 403.8296509, -103.9003830, 396.2039795, -502.4128723, 507.7300415
4: -90.6926956, 367.2429504, -88.7171707, 360.1357422, -450.8284302, 455.9601135

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8020678, upper bound: 339.8033669
time: 1.06 seconds

## Relational analysis of NS_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8020678, upper bound: 339.8047698
time: 1.10 seconds

## BFS NS instance: NS_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -84.0302887, 281.8778992, -365.1642456, 363.3833008
1: -116.9414825, 277.0936890, -118.0316544, 279.7136536, -396.6551514, 395.1253357
2: -99.1595917, 305.1754456, -100.0824814, 308.0635986, -407.2231445, 405.2578735
3: -104.0396271, 396.9595032, -104.9687729, 400.3738098, -504.4134521, 501.9282227
4: -88.8424301, 360.8237305, -89.6207352, 363.9067383, -452.7491760, 450.4444580

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A2_A1_B1

### Relational analysis result of NS_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967397, upper bound: 339.7961309
time: 1.03 seconds

## Relational analysis of NS_B1_B2_A2_A1_B2

### Relational analysis result of NS_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967397, upper bound: 339.8005007
time: 0.85 seconds

## BFS NS instance: NS_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -84.1043701, 282.1285706, -366.7696228, 368.3857727
1: -118.8840942, 281.9696045, -118.1360779, 279.9616394, -398.8457336, 400.1056213
2: -100.8020096, 310.5105286, -100.1703873, 308.3364563, -409.1384583, 410.6809082
3: -105.7517395, 403.7609863, -105.0615692, 400.7293396, -506.4810181, 508.8225708
4: -90.2919159, 366.9395752, -89.6990509, 364.2296143, -454.5215149, 456.6386108

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A2_A2_B1

### Relational analysis result of NS_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8033462, upper bound: 339.8040320
time: 0.72 seconds

## Relational analysis of NS_B1_B2_A2_A2_B2

### Relational analysis result of NS_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8032955, upper bound: 339.8029687
time: 0.93 seconds

## BFS NS instance: NS_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -85.0222626, 283.9200745, -82.3742294, 274.9334106, -359.9556885, 366.2943115
1: -119.2890396, 281.8406067, -115.5383682, 272.8794861, -392.1684570, 397.3789673
2: -101.1887512, 310.4375000, -98.0269623, 300.6091309, -401.7978821, 408.4643860
3: -106.1172714, 403.4742737, -102.7902451, 390.7839355, -496.9012146, 506.2645264
4: -90.6156235, 366.9209900, -87.8121414, 355.4825745, -446.0982056, 454.7330933

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8035604
time: 0.97 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8035604
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -83.8897476, 280.0026550, -365.0975647, 368.0602112
1: -119.3920059, 282.0884705, -117.6831970, 277.9667664, -397.3587036, 399.7716675
2: -101.2751236, 310.7092590, -99.8349075, 306.1901550, -407.4652405, 410.5441589
3: -106.2089005, 403.8296509, -104.6845016, 397.9117126, -504.1206055, 508.5141602
4: -90.6926956, 367.2429504, -89.4048462, 361.9069214, -452.5996094, 456.6477966

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8049406
time: 0.76 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8049862
time: 0.97 seconds

## BFS NS instance: NS_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -85.0222626, 283.9200745, -82.9877167, 278.3031006, -363.3253784, 366.9077759
1: -119.2890396, 281.8406067, -116.5166245, 276.0668640, -395.3558655, 398.3572388
2: -101.1887512, 310.4375000, -98.8013992, 304.0500183, -405.2387695, 409.2388611
3: -106.1172714, 403.4742737, -103.6618652, 395.4787903, -501.5960693, 507.1361389
4: -90.6156235, 366.9209900, -88.5221558, 359.4860229, -450.1016235, 455.4431458

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8007783, upper bound: 339.7954857
time: 1.00 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023517, upper bound: 339.7976404
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -85.0949707, 284.1704712, -84.2925873, 283.0604248, -368.1553345, 368.4630737
1: -119.3920059, 282.0884705, -118.3896637, 280.7737732, -400.1657104, 400.4781494
2: -101.2751236, 310.7092590, -100.3853149, 309.1988220, -410.4739075, 411.0945740
3: -106.2089005, 403.8296509, -105.3119278, 402.0356445, -508.2445374, 509.1415710
4: -90.6926956, 367.2429504, -89.9195251, 365.3794861, -456.0721741, 457.1624756

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040684, upper bound: 339.8039865
time: 0.90 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023517, upper bound: 339.8042324
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -84.6512070, 282.6155396, -365.9018860, 364.0041504
1: -116.9414825, 277.0936890, -118.7632294, 280.5619507, -397.5034180, 395.8569336
2: -99.1595917, 305.1754456, -100.7468948, 309.0411072, -408.2006531, 405.9223328
3: -104.0396271, 396.9595032, -105.6491394, 401.6319275, -505.6715393, 502.6086426
4: -88.8424301, 360.8237305, -90.2195511, 365.2533569, -454.0957947, 451.0432739

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954857, upper bound: 339.8007783
time: 0.70 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976404, upper bound: 339.8023517
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -84.7256393, 282.8713074, -367.5123291, 369.0070496
1: -118.8840942, 281.9696045, -118.8686752, 280.8153687, -399.6994629, 400.8382568
2: -100.8020096, 310.5105286, -100.8353577, 309.3188782, -410.1208801, 411.3458862
3: -105.7517395, 403.7609863, -105.7429886, 401.9952698, -507.7469482, 509.5039673
4: -90.2919159, 366.9395752, -90.2985229, 365.5823364, -455.8742371, 457.2380676

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039682, upper bound: 339.8022108
time: 0.74 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8039682, upper bound: 339.8040898
time: 1.28 seconds

## BFS NS instance: NS_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -85.4344482, 286.9944458, -82.9877167, 278.3031006, -363.7375488, 369.9821472
1: -120.0090866, 284.6630859, -116.5166245, 276.0668640, -396.0759583, 401.1797180
2: -101.7500534, 313.4707031, -98.8013992, 304.0500183, -405.8000793, 412.2720642
3: -106.7556610, 407.6220093, -103.6618652, 395.4787903, -502.2344360, 511.2838745
4: -91.1380386, 370.4396362, -88.5221558, 359.4860229, -450.6239929, 458.9617920

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967462, upper bound: 339.7967462
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967462, upper bound: 339.7969932
time: 0.90 seconds

## BFS NS instance: NS_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -85.5067825, 287.2420959, -84.2925873, 283.0604248, -368.5671997, 371.5346375
1: -120.1114273, 284.9082947, -118.3896637, 280.7737732, -400.8851318, 403.2979736
2: -101.8361511, 313.7402954, -100.3853149, 309.1988220, -411.0349731, 414.1256104
3: -106.8467102, 407.9730225, -105.3119278, 402.0356445, -508.8823547, 513.2849731
4: -91.2148666, 370.7578735, -89.9195251, 365.3794861, -456.5943604, 460.6773682

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8038931, upper bound: 339.8033183
time: 0.97 seconds

## Relational analysis of NS_B2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017825, upper bound: 339.8033027
time: 0.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.68 seconds
NS_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8012659
NS_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8042104
NS_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8048113, upper bound: 339.8012659
NS_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8048113, upper bound: 339.8049209
NS_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7622247, upper bound: 339.7790232
NS_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7965194, upper bound: 339.8004185
NS_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8038332, upper bound: 339.8040684
NS_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7973500, upper bound: 339.8029820
NS_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8023026, upper bound: 339.7962509
NS_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8026643, upper bound: 339.7975155
NS_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8020678, upper bound: 339.8033669
NS_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8020678, upper bound: 339.8047698
NS_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7967397, upper bound: 339.7961309
NS_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7967397, upper bound: 339.8005007
NS_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8033462, upper bound: 339.8040320
NS_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8032955, upper bound: 339.8029687
NS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8035604
NS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8035604
NS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8049406
NS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8035604, upper bound: 339.8049862
NS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8007783, upper bound: 339.7954857
NS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8023517, upper bound: 339.7976404
NS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8040684, upper bound: 339.8039865
NS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8023517, upper bound: 339.8042324
NS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7954857, upper bound: 339.8007783
NS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7976404, upper bound: 339.8023517
NS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8039682, upper bound: 339.8022108
NS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8039682, upper bound: 339.8040898
NS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7967462, upper bound: 339.7967462
NS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.7967462, upper bound: 339.7969932
NS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8038931, upper bound: 339.8033183
NS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -339.8017825, upper bound: 339.8033027

## BFS NS instance: NS_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -81.4270630, 270.8995056, -353.5843201, 357.4660339
1: -115.9796906, 273.9595032, -114.1825333, 269.1073914, -385.0870361, 388.1420288
2: -98.3984375, 301.7886658, -96.8973541, 296.5032349, -394.9016724, 398.6860046
3: -103.1827164, 392.3468323, -101.5671158, 385.0579529, -488.2406616, 493.9139404
4: -88.1445007, 356.8894043, -86.7820969, 350.4340515, -438.5785522, 443.6715088

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8006197, upper bound: 339.7915148
time: 1.05 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8013656
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -83.0612259, 276.6855164, -359.3703003, 359.1001892
1: -115.9796906, 273.9595032, -116.4843903, 274.8233948, -390.8030701, 390.4439087
2: -98.3984375, 301.7886658, -98.8284225, 302.7738953, -401.1723328, 400.6170654
3: -103.1827164, 392.3468323, -103.5972824, 393.0469360, -496.2296448, 495.9441223
4: -88.1445007, 356.8894043, -88.4753418, 357.5619202, -445.7064209, 445.3647156

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8006197, upper bound: 339.8042104
time: 0.87 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8038288
time: 1.33 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -81.4270630, 270.8995056, -355.1339111, 362.6503906
1: -118.1718445, 279.1607971, -114.1825333, 269.1073914, -387.2792358, 393.3433228
2: -100.2456055, 307.4933777, -96.8973541, 296.5032349, -396.7488403, 404.3906860
3: -105.1194687, 399.6337585, -101.5671158, 385.0579529, -490.1774292, 501.2008667
4: -89.7725525, 363.4640198, -86.7820969, 350.4340515, -440.2066040, 450.2460938

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019954, upper bound: 339.7915157
time: 1.15 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8046545, upper bound: 339.8012659
time: 1.03 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -83.0612259, 276.6855164, -360.9198914, 364.2845459
1: -118.1718445, 279.1607971, -116.4843903, 274.8233948, -392.9952393, 395.6452026
2: -100.2456055, 307.4933777, -98.8284225, 302.7738953, -403.0195007, 406.3218079
3: -105.1194687, 399.6337585, -103.5972824, 393.0469360, -498.1664124, 503.2310486
4: -89.7725525, 363.4640198, -88.4753418, 357.5619202, -447.3344727, 451.9393005

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019954, upper bound: 339.8043922
time: 0.89 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8046545, upper bound: 339.8035890
time: 1.11 seconds

## BFS NS instance: NS_B1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -81.4670258, 274.0292664, -83.8959808, 279.6270752, -361.0940857, 357.9252319
1: -114.2544403, 271.7889099, -117.6765442, 277.7276001, -391.9820557, 389.4654541
2: -96.9108582, 299.3471985, -99.8341293, 305.9558411, -402.8666992, 399.1813049
3: -101.6959457, 389.3615417, -104.6593323, 397.2304382, -498.9263916, 494.0208740
4: -86.8806610, 353.7731018, -89.3746719, 361.3166809, -448.1973267, 443.1477661

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_A1_A1_B1

### Relational analysis result of NS_B1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7622133, upper bound: 339.7755512
time: 0.70 seconds

## Relational analysis of NS_B1_B1_A2_A1_A1_B2

### Relational analysis result of NS_B1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7622133, upper bound: 339.7790232
time: 1.17 seconds

## BFS NS instance: NS_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -83.0684814, 278.6816711, -83.8959808, 279.6270752, -362.6955566, 362.5776062
1: -116.6364975, 276.4191284, -117.6765442, 277.7276001, -394.3641052, 394.0956726
2: -98.9037476, 304.4310913, -99.8341293, 305.9558411, -404.8595886, 404.2652283
3: -103.7691269, 396.0045166, -104.6593323, 397.2304382, -500.9995728, 500.6638489
4: -88.6151886, 359.9463806, -89.3746719, 361.3166809, -449.9318848, 449.3210449

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_A1_A2_B1

### Relational analysis result of NS_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965194, upper bound: 339.7976078
time: 0.83 seconds

## Relational analysis of NS_B1_B1_A2_A1_A2_B2

### Relational analysis result of NS_B1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965194, upper bound: 339.8004185
time: 0.70 seconds

## BFS NS instance: NS_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -84.0325317, 282.1983337, -82.7144318, 275.3447876, -359.3773193, 364.9127502
1: -118.0285568, 279.9098511, -116.0173721, 273.4808350, -391.5093384, 395.9272156
2: -100.0767822, 308.2475281, -98.4105682, 301.2718811, -401.3486633, 406.6580811
3: -104.9913025, 400.8173523, -103.1867676, 391.1961975, -496.1875000, 504.0041199
4: -89.6422729, 364.2773132, -88.1052322, 355.9544067, -445.5966492, 452.3825378

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017501, upper bound: 339.7906850
time: 1.25 seconds

## Relational analysis of NS_B1_B1_A2_A2_B1_B2

### Relational analysis result of NS_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017501, upper bound: 339.8040532
time: 1.02 seconds

## BFS NS instance: NS_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -83.3901596, 277.9269104, -362.5679016, 367.6715698
1: -118.8840942, 281.9696045, -116.9604645, 276.0585327, -394.9426270, 398.9300537
2: -100.8020096, 310.5105286, -99.2210312, 304.1276855, -404.9296875, 409.7315674
3: -105.7517395, 403.7609863, -104.0260696, 394.8450012, -500.5966797, 507.7870483
4: -90.2919159, 366.9395752, -88.8294754, 359.1751709, -449.4670715, 455.7690125

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040606, upper bound: 339.8000571
time: 1.03 seconds

## Relational analysis of NS_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8040606, upper bound: 339.8029820
time: 0.89 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -83.7774048, 279.6228027, -81.1385422, 271.6237488, -355.4011230, 360.7613525
1: -117.5312729, 277.5732422, -113.9240875, 269.5771484, -387.1084290, 391.4972839
2: -99.6833115, 305.7511597, -96.6205215, 296.9439087, -396.6272278, 402.3716431
3: -104.5612564, 397.3877869, -101.3312683, 385.9013977, -490.4626160, 498.7190552
4: -89.2740173, 361.4435120, -86.5535126, 350.8887634, -440.1627808, 447.9970093

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B2_A1_B1_A1_A1

### Relational analysis result of NS_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985568, upper bound: 339.7949967
time: 0.94 seconds

## Relational analysis of NS_B1_B2_A1_B1_A1_A2

### Relational analysis result of NS_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023026, upper bound: 339.7962509
time: 1.26 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -84.4434967, 281.9713440, -81.6958694, 273.5408630, -357.9843750, 363.6672058
1: -118.4702377, 279.9180603, -114.7062836, 271.4855042, -389.9557495, 394.6242981
2: -100.4890213, 308.3388977, -97.2843628, 299.0462952, -399.5353088, 405.6232605
3: -105.3921738, 400.7395325, -102.0274506, 388.6209106, -494.0130615, 502.7669678
4: -89.9932022, 364.4594116, -87.1474457, 353.3419495, -443.3350525, 451.6068726

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8012878, upper bound: 339.7975155
time: 1.16 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8012878, upper bound: 339.7975154
time: 0.91 seconds

## BFS NS instance: NS_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -83.1899185, 278.9436035, -361.6284180, 359.2289124
1: -115.9796906, 273.9595032, -116.8321838, 276.8170471, -392.7967224, 390.7916870
2: -98.3984375, 301.7886658, -99.0710449, 304.8868408, -403.2852783, 400.8596191
3: -103.1827164, 392.3468323, -103.9003830, 396.2039795, -499.3866882, 496.2472229
4: -88.1445007, 356.8894043, -88.7171707, 360.1357422, -448.2802429, 445.6065674

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A1_B2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7363138, upper bound: 339.7529662
time: 0.74 seconds

## Relational analysis of NS_B1_B2_A1_B2_A1_B2

### Relational analysis result of NS_B1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7356603, upper bound: 339.7543190
time: 0.86 seconds

## BFS NS instance: NS_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -83.1899185, 278.9436035, -363.1780090, 364.4132385
1: -118.1718445, 279.1607971, -116.8321838, 276.8170471, -394.9888916, 395.9929810
2: -100.2456055, 307.4933777, -99.0710449, 304.8868408, -405.1324463, 406.5643311
3: -105.1194687, 399.6337585, -103.9003830, 396.2039795, -501.3234558, 503.5341492
4: -89.7725525, 363.4640198, -88.7171707, 360.1357422, -449.9082947, 452.1811829

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7363138, upper bound: 339.7752806
time: 1.11 seconds

## Relational analysis of NS_B1_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7356603, upper bound: 339.7665969
time: 0.77 seconds

## BFS NS instance: NS_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -81.6958694, 273.5408630, -356.8272095, 361.0488586
1: -116.9414825, 277.0936890, -114.7062836, 271.4855042, -388.4270020, 391.7999878
2: -99.1595917, 305.1754456, -97.2843628, 299.0462952, -398.2058105, 402.4598083
3: -104.0396271, 396.9595032, -102.0274506, 388.6209106, -492.6605225, 498.9869385
4: -88.8424301, 360.8237305, -87.1474457, 353.3419495, -442.1843872, 447.9711914

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A2_A1_B1_A1

### Relational analysis result of NS_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7609477, upper bound: 339.7716440
time: 0.83 seconds

## Relational analysis of NS_B1_B2_A2_A1_B1_A2

### Relational analysis result of NS_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7961801, upper bound: 339.7951241
time: 0.86 seconds

## BFS NS instance: NS_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -83.1899185, 278.9436035, -362.2299500, 362.5429077
1: -116.9414825, 277.0936890, -116.8321838, 276.8170471, -393.7585449, 393.9258728
2: -99.1595917, 305.1754456, -99.0710449, 304.8868408, -404.0463867, 404.2463989
3: -104.0396271, 396.9595032, -103.9003830, 396.2039795, -500.2435913, 500.8598938
4: -88.8424301, 360.8237305, -88.7171707, 360.1357422, -448.9781799, 449.5408936

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7609477, upper bound: 339.7779780
time: 0.98 seconds

## Relational analysis of NS_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7961801, upper bound: 339.7998889
time: 1.06 seconds

## BFS NS instance: NS_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -84.0325317, 282.1983337, -82.6777344, 276.9024353, -360.9349670, 364.8760681
1: -118.0285568, 279.9098511, -116.1024857, 274.8269348, -392.8554688, 396.0122986
2: -100.0767822, 308.2475281, -98.4431381, 302.6804199, -402.7572021, 406.6906128
3: -104.9913025, 400.8173523, -103.2545929, 393.3429260, -498.3342285, 504.0719299
4: -89.6422729, 364.2773132, -88.1485748, 357.6156311, -447.2578735, 452.4258728

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7999939, upper bound: 339.8017046
time: 0.88 seconds

## Relational analysis of NS_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
time: 0.88 seconds

## BFS NS instance: NS_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -83.5397720, 280.2258301, -364.8668518, 367.8211670
1: -118.8840942, 281.9696045, -117.3339691, 278.0932312, -396.9773254, 399.3035889
2: -100.8020096, 310.5105286, -99.4854889, 306.2893066, -407.0913086, 409.9960327
3: -105.7517395, 403.7609863, -104.3522949, 398.0578918, -503.8096008, 508.1132812
4: -90.2919159, 366.9395752, -89.0907593, 361.8130798, -452.1049805, 456.0303345

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021021, upper bound: 339.7967223
time: 0.93 seconds

## Relational analysis of NS_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8021021, upper bound: 339.8029687
time: 1.07 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -82.3742294, 274.9334106, -357.6182251, 358.4131775
1: -115.9796906, 273.9595032, -115.5383682, 272.8794861, -388.8591614, 389.4978638
2: -98.3984375, 301.7886658, -98.0269623, 300.6091309, -399.0075684, 399.8155823
3: -103.1827164, 392.3468323, -102.7902451, 390.7839355, -493.9666443, 495.1370544
4: -88.1445007, 356.8894043, -87.8121414, 355.4825745, -443.6270752, 444.7014771

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7913879, upper bound: 339.7715440
time: 1.29 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_B2

### Relational analysis result of NS_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8033021, upper bound: 339.8030696
time: 1.13 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -82.3742294, 274.9334106, -359.1678467, 363.5975647
1: -118.1718445, 279.1607971, -115.5383682, 272.8794861, -391.0513306, 394.6991577
2: -100.2456055, 307.4933777, -98.0269623, 300.6091309, -400.8547363, 405.5203247
3: -105.1194687, 399.6337585, -102.7902451, 390.7839355, -495.9034119, 502.4239807
4: -89.7725525, 363.4640198, -87.8121414, 355.4825745, -445.2551270, 451.2760620

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7719419, upper bound: 339.7894140
time: 1.06 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8033020, upper bound: 339.8030697
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -83.8897476, 280.0026550, -362.6874695, 359.9286804
1: -115.9796906, 273.9595032, -117.6831970, 277.9667664, -393.9464111, 391.6427002
2: -98.3984375, 301.7886658, -99.8349075, 306.1901550, -404.5885925, 401.6235657
3: -103.1827164, 392.3468323, -104.6845016, 397.9117126, -501.0944214, 497.0313416
4: -88.1445007, 356.8894043, -89.4048462, 361.9069214, -450.0514221, 446.2942505

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7387205, upper bound: 339.7628445
time: 1.33 seconds

## Relational analysis of NS_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7359153, upper bound: 339.7569537
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -83.8897476, 280.0026550, -364.2370605, 365.1130676
1: -118.1718445, 279.1607971, -117.6831970, 277.9667664, -396.1386108, 396.8439941
2: -100.2456055, 307.4933777, -99.8349075, 306.1901550, -406.4357605, 407.3282776
3: -105.1194687, 399.6337585, -104.6845016, 397.9117126, -503.0311890, 504.3182678
4: -89.7725525, 363.4640198, -89.4048462, 361.9069214, -451.6794739, 452.8688354

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7387205, upper bound: 339.7816526
time: 1.06 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7359153, upper bound: 339.7359153
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -83.7774048, 279.6228027, -82.4264603, 276.3891296, -360.1665039, 362.0492554
1: -117.5312729, 277.5732422, -115.7275696, 274.1688843, -391.7001648, 393.3008118
2: -99.6833115, 305.7511597, -98.1327286, 301.9580078, -401.6413269, 403.8838806
3: -104.5612564, 397.3877869, -102.9597626, 392.7521973, -497.3134460, 500.3475342
4: -89.2740173, 361.4435120, -87.9236145, 357.0223083, -446.2963257, 449.3671265

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8007783, upper bound: 339.7954857
time: 0.75 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8002256, upper bound: 339.7953436
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -84.4434967, 281.9713440, -82.9877167, 278.3031006, -362.7465820, 364.9590149
1: -118.4702377, 279.9180603, -116.5166245, 276.0668640, -394.5371094, 396.4346924
2: -100.4890213, 308.3388977, -98.8013992, 304.0500183, -404.5390320, 407.1402893
3: -105.3921738, 400.7395325, -103.6618652, 395.4787903, -500.8709106, 504.4013977
4: -89.9932022, 364.4594116, -88.5221558, 359.4860229, -449.4791260, 452.9815674

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009855, upper bound: 339.7976404
time: 1.06 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009855, upper bound: 339.7976404
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -83.8497238, 279.8697510, -83.6860199, 280.9848938, -364.8346252, 363.5556946
1: -117.6336365, 277.8181152, -117.5372162, 278.7214661, -396.3550720, 395.3552551
2: -99.7692108, 306.0202332, -99.6626129, 306.9438171, -406.7130127, 405.6828613
3: -104.6523438, 397.7386169, -104.5541153, 399.1029968, -503.7553101, 502.2927246
4: -89.3506927, 361.7621155, -89.2721252, 362.7269897, -452.0776978, 451.0342407

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_B2_A1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7987364, upper bound: 339.7952390
time: 1.01 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B2_A1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967455, upper bound: 339.8010276
time: 0.86 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035764, upper bound: 339.8035671
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -84.5164337, 282.2228394, -84.2925873, 283.0604248, -367.5768127, 366.5153503
1: -118.5735626, 280.1671753, -118.3896637, 280.7737732, -399.3473511, 398.5568237
2: -100.5756989, 308.6119080, -100.3853149, 309.1988220, -409.7745056, 408.9972229
3: -105.4840927, 401.0967712, -105.3119278, 402.0356445, -507.5197449, 506.4086914
4: -90.0705109, 364.7825928, -89.9195251, 365.3794861, -455.4500122, 454.7021179

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009855, upper bound: 339.8039682
time: 1.17 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8009855, upper bound: 339.8037018
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -82.7254562, 277.4403381, -83.4158401, 278.3565674, -361.0820312, 360.8561401
1: -116.1529312, 275.1970215, -117.0192337, 276.3326111, -392.4855347, 392.2162476
2: -98.4912949, 303.0850525, -99.2529678, 304.3892517, -402.8804932, 402.3379211
3: -103.3379898, 394.2349854, -104.1055527, 395.6005554, -498.9385376, 498.3405457
4: -88.2442398, 358.3619385, -88.8883133, 359.8280029, -448.0722351, 447.2502441

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A1_B1_B1

### Relational analysis result of NS_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937009, upper bound: 339.7991011
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B1_A1_B1_B2

### Relational analysis result of NS_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7939909, upper bound: 339.7990870
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -84.0725250, 280.6664124, -363.9527588, 363.4255371
1: -116.9414825, 277.0936890, -117.9444351, 278.6379700, -395.5794678, 395.0381165
2: -99.1595917, 305.1754456, -100.0471802, 306.9411621, -406.1006470, 405.2226257
3: -104.0396271, 396.9595032, -104.9240875, 398.8946533, -502.9342651, 501.8835754
4: -88.8424301, 360.8237305, -89.5971909, 362.7905884, -451.6330261, 450.4209290

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B1_A1_B2_B1

### Relational analysis result of NS_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976404, upper bound: 339.8009855
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2_B2

### Relational analysis result of NS_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976404, upper bound: 339.8023517
time: 1.03 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -82.3742294, 274.9334106, -359.5744324, 366.6555786
1: -118.8840942, 281.9696045, -115.5383682, 272.8794861, -391.7635803, 397.5079651
2: -100.8020096, 310.5105286, -98.0269623, 300.6091309, -401.4111328, 408.5374756
3: -105.7517395, 403.7609863, -102.7902451, 390.7839355, -496.5356445, 506.5511780
4: -90.2919159, 366.9395752, -87.8121414, 355.4825745, -445.7744751, 454.7516479

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7640854, upper bound: 339.7843640
time: 1.16 seconds

## Relational analysis of NS_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035932, upper bound: 339.8017190
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -83.8897476, 280.0026550, -364.6436462, 368.1710815
1: -118.8840942, 281.9696045, -117.6831970, 277.9667664, -396.8508301, 399.6528015
2: -100.8020096, 310.5105286, -99.8349075, 306.1901550, -406.9921570, 410.3454285
3: -105.7517395, 403.7609863, -104.6845016, 397.9117126, -503.6634216, 508.4454956
4: -90.2919159, 366.9395752, -89.4048462, 361.9069214, -452.1988220, 456.3444214

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7583148, upper bound: 339.7658726
time: 1.03 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7556634, upper bound: 339.7364512
time: 1.10 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -82.9877167, 278.3031006, -361.5894470, 362.3406982
1: -116.9414825, 277.0936890, -116.5166245, 276.0668640, -393.0083618, 393.6103210
2: -99.1595917, 305.1754456, -98.8013992, 304.0500183, -403.2095947, 403.9768372
3: -104.0396271, 396.9595032, -103.6618652, 395.4787903, -499.5183716, 500.6213074
4: -88.8424301, 360.8237305, -88.5221558, 359.4860229, -448.3284607, 449.3458862

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B2_B1_A1_B1

### Relational analysis result of NS_B2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7799380, upper bound: 339.7610208
time: 0.93 seconds

## Relational analysis of NS_B2_A2_B2_B1_A1_B2

### Relational analysis result of NS_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962465, upper bound: 339.7962465
time: 0.99 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -82.9877167, 278.3031006, -362.9441223, 367.2690735
1: -118.8840942, 281.9696045, -116.5166245, 276.0668640, -394.9509583, 398.4862061
2: -100.8020096, 310.5105286, -98.8013992, 304.0500183, -404.8520203, 409.3119202
3: -105.7517395, 403.7609863, -103.6618652, 395.4787903, -501.2304688, 507.4228210
4: -90.2919159, 366.9395752, -88.5221558, 359.4860229, -449.7778931, 455.4617310

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B2_B1_A2_B1

### Relational analysis result of NS_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7799380, upper bound: 339.7611857
time: 0.94 seconds

## Relational analysis of NS_B2_A2_B2_B1_A2_B2

### Relational analysis result of NS_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962465, upper bound: 339.7964935
time: 0.99 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -84.0918808, 282.1043091, -83.6860199, 280.9848938, -365.0767822, 365.7903137
1: -118.0949783, 279.8674011, -117.5372162, 278.7214661, -396.8164368, 397.4046021
2: -100.1274261, 308.2096863, -99.6626129, 306.9438171, -407.0712280, 407.8723145
3: -105.0574951, 400.7886963, -104.5541153, 399.1029968, -504.1604919, 505.3428040
4: -89.6795883, 364.3168945, -89.2721252, 362.7269897, -452.4065552, 453.5890198

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_B2_A1_A1

### Relational analysis result of NS_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8015469, upper bound: 339.7999575
time: 0.92 seconds

## Relational analysis of NS_B2_A2_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8014011, upper bound: 339.8025844
time: 0.98 seconds

## Relational analysis of NS_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -84.9450150, 285.3491821, -84.2925873, 283.0604248, -368.0054321, 369.6416931
1: -119.3145599, 283.0417786, -118.3896637, 280.7737732, -400.0882568, 401.4314270
2: -101.1553192, 311.6925049, -100.3853149, 309.1988220, -410.3540955, 412.0778198
3: -106.1418076, 405.3186340, -105.3119278, 402.0356445, -508.1774597, 510.6305542
4: -90.6101379, 368.3550720, -89.9195251, 365.3794861, -455.9896240, 458.2745972

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_B2_B2_A2_A1

### Relational analysis result of NS_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967456, upper bound: 339.8017825
time: 0.96 seconds

## Relational analysis of NS_B2_A2_B2_B2_A2_A2

### Relational analysis result of NS_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967456, upper bound: 339.8033027
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.85 seconds
NS_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8006197, upper bound: 339.7915148
NS_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8013656
NS_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8006197, upper bound: 339.8042104
NS_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8032543, upper bound: 339.8038288
NS_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8019954, upper bound: 339.7915157
NS_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8046545, upper bound: 339.8012659
NS_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8019954, upper bound: 339.8043922
NS_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8046545, upper bound: 339.8035890
NS_B1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7622133, upper bound: 339.7755512
NS_B1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7622133, upper bound: 339.7790232
NS_B1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7965194, upper bound: 339.7976078
NS_B1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7965194, upper bound: 339.8004185
NS_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8017501, upper bound: 339.7906850
NS_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8017501, upper bound: 339.8040532
NS_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8040606, upper bound: 339.8000571
NS_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8040606, upper bound: 339.8029820
NS_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7985568, upper bound: 339.7949967
NS_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8023026, upper bound: 339.7962509
NS_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8012878, upper bound: 339.7975155
NS_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8012878, upper bound: 339.7975154
NS_B1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7363138, upper bound: 339.7529662
NS_B1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7356603, upper bound: 339.7543190
NS_B1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7363138, upper bound: 339.7752806
NS_B1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7356603, upper bound: 339.7665969
NS_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7609477, upper bound: 339.7716440
NS_B1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7961801, upper bound: 339.7951241
NS_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7609477, upper bound: 339.7779780
NS_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7961801, upper bound: 339.7998889
NS_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7999939, upper bound: 339.8017046
NS_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
NS_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8021021, upper bound: 339.7967223
NS_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8021021, upper bound: 339.8029687
NS_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7913879, upper bound: 339.7715440
NS_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8033021, upper bound: 339.8030696
NS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7719419, upper bound: 339.7894140
NS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8033020, upper bound: 339.8030697
NS_B2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7387205, upper bound: 339.7628445
NS_B2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7359153, upper bound: 339.7569537
NS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7387205, upper bound: 339.7816526
NS_B2_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7359153, upper bound: 339.7359153
NS_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8007783, upper bound: 339.7954857
NS_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8002256, upper bound: 339.7953436
NS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8009855, upper bound: 339.7976404
NS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8009855, upper bound: 339.7976404
NS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7967455, upper bound: 339.8010276
NS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8035764, upper bound: 339.8035671
NS_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8009855, upper bound: 339.8039682
NS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8009855, upper bound: 339.8037018
NS_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7937009, upper bound: 339.7991011
NS_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7939909, upper bound: 339.7990870
NS_B2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7976404, upper bound: 339.8009855
NS_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7976404, upper bound: 339.8023517
NS_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7640854, upper bound: 339.7843640
NS_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8035932, upper bound: 339.8017190
NS_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7583148, upper bound: 339.7658726
NS_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7556634, upper bound: 339.7364512
NS_B2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7799380, upper bound: 339.7610208
NS_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7962465, upper bound: 339.7962465
NS_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7799380, upper bound: 339.7611857
NS_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7962465, upper bound: 339.7964935
NS_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8014011, upper bound: 339.8025844
NS_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
NS_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7967456, upper bound: 339.8017825
NS_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 0, lower bound: -339.7967456, upper bound: 339.8033027

## BFS NS instance: NS_B1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -82.1529694, 274.2243652, -80.4826736, 267.5901794, -349.7431641, 354.7070312
1: -115.2323303, 272.1658936, -112.8496170, 265.7628174, -380.9951477, 385.0154419
2: -97.7629929, 299.8114929, -95.7479324, 292.7885742, -390.5515442, 395.5593872
3: -102.5186081, 389.7501221, -100.3842545, 380.3419189, -482.8605347, 490.1343079
4: -87.5767365, 354.5466919, -85.7559891, 346.1466980, -433.7233887, 440.3026733

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985034, upper bound: 339.7895439
time: 0.89 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8000652, upper bound: 339.7904316
time: 1.07 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -80.8200302, 268.8648682, -351.5496826, 356.8590393
1: -115.9796906, 273.9595032, -113.3218536, 267.1051025, -383.0847778, 387.2813416
2: -98.3984375, 301.7886658, -96.1628876, 294.3084106, -392.7068481, 397.9515381
3: -103.1827164, 392.3468323, -100.8058624, 382.2045288, -485.3872375, 493.1527100
4: -88.1445007, 356.8894043, -86.1291962, 347.8483276, -435.9928284, 443.0186157

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A1_A1_B1_B2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7907824, upper bound: 339.7695433
time: 0.89 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8027122, upper bound: 339.8003615
time: 0.95 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -82.1529694, 274.2243652, -81.8182907, 272.2176514, -354.3706055, 356.0426331
1: -115.2323303, 272.1658936, -114.7416687, 270.3858032, -385.6181335, 386.9075317
2: -97.7629929, 299.8114929, -97.3347473, 297.8790894, -395.6420593, 397.1462402
3: -102.5186081, 389.7501221, -102.0502548, 386.7622681, -489.2808838, 491.8003845
4: -87.5767365, 354.5466919, -87.1439819, 351.9501038, -439.5268250, 441.6906433

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7996897, upper bound: 339.8007935
time: 0.87 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8019150, upper bound: 339.8036706
time: 1.04 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -82.4752197, 274.7119751, -357.3967896, 358.5142212
1: -115.9796906, 273.9595032, -115.6550064, 272.8845825, -388.8642578, 389.6144714
2: -98.3984375, 301.7886658, -98.1201096, 300.6493835, -399.0478210, 399.9087830
3: -103.1827164, 392.3468323, -102.8633804, 390.2746582, -493.4573669, 495.2102051
4: -88.1445007, 356.8894043, -87.8455429, 355.0695496, -443.2140503, 444.7349548

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7376770, upper bound: 339.7548072
time: 0.93 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7371150, upper bound: 339.7547895
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -83.6280212, 279.1757202, -80.4826736, 267.5901794, -351.2182007, 359.6583862
1: -117.3192368, 277.1330872, -112.8496170, 265.7628174, -383.0820618, 389.9826050
2: -99.5217514, 305.2602234, -95.7479324, 292.7885742, -392.3102722, 401.0080872
3: -104.3624115, 396.7291565, -100.3842545, 380.3419189, -484.7043152, 497.1133423
4: -89.1260223, 360.8087769, -85.7559891, 346.1466980, -435.2727051, 446.5647583

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7983758, upper bound: 339.7894986
time: 1.16 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017527, upper bound: 339.7904214
time: 1.04 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -80.8200302, 268.8648682, -353.0993042, 362.0433655
1: -118.1718445, 279.1607971, -113.3218536, 267.1051025, -385.2769470, 392.4826660
2: -100.2456055, 307.4933777, -96.1628876, 294.3084106, -394.5540161, 403.6562500
3: -105.1194687, 399.6337585, -100.8058624, 382.2045288, -487.3240051, 500.4396362
4: -89.7725525, 363.4640198, -86.1291962, 347.8483276, -437.6208801, 449.5932007

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7710511, upper bound: 339.7811421
time: 0.95 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042069, upper bound: 339.8002647
time: 0.72 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -83.6280212, 279.1757202, -81.8182907, 272.2176514, -355.8456421, 360.9939880
1: -117.3192368, 277.1330872, -114.7416687, 270.3858032, -387.7050476, 391.8747253
2: -99.5217514, 305.2602234, -97.3347473, 297.8790894, -397.4007874, 402.5949707
3: -104.3624115, 396.7291565, -102.0502548, 386.7622681, -491.1246948, 498.7794189
4: -89.1260223, 360.8087769, -87.1439819, 351.9501038, -441.0761108, 447.9527283

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A2_B2_B1_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8005084, upper bound: 339.8020825
time: 1.15 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985200, upper bound: 339.8004114
time: 1.02 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8038537, upper bound: 339.8038996
time: 0.96 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -84.2344208, 281.2233276, -82.4752197, 274.7119751, -358.9464111, 363.6985474
1: -118.1718445, 279.1607971, -115.6550064, 272.8845825, -391.0564270, 394.8157959
2: -100.2456055, 307.4933777, -98.1201096, 300.6493835, -400.8949890, 405.6134949
3: -105.1194687, 399.6337585, -102.8633804, 390.2746582, -495.3941345, 502.4971313
4: -89.7725525, 363.4640198, -87.8455429, 355.0695496, -444.8421021, 451.3095398

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A2_B2_B2_B1

### Relational analysis result of NS_B1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7672659, upper bound: 339.7757611
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A1_A2_B2_B2_B2

### Relational analysis result of NS_B1_B1_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7647153, upper bound: 339.7643158
time: 1.29 seconds

## BFS NS instance: NS_B1_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -81.4670258, 274.0292664, -81.4270630, 270.8995056, -352.3665161, 355.4563293
1: -114.2544403, 271.7889099, -114.1825333, 269.1073914, -383.3618164, 385.9714355
2: -96.9108582, 299.3471985, -96.8973541, 296.5032349, -393.4140930, 396.2445374
3: -101.6959457, 389.3615417, -101.5671158, 385.0579529, -486.7539062, 490.9286194
4: -86.8806610, 353.7731018, -86.7820969, 350.4340515, -437.3146973, 440.5552063

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_A1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7427272, upper bound: 339.7305326
time: 1.31 seconds

## Relational analysis of NS_B1_B1_A2_A1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7211445, upper bound: 339.7281173
time: 0.99 seconds

## BFS NS instance: NS_B1_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -81.4670258, 274.0292664, -83.0612259, 276.6855164, -358.1524963, 357.0904846
1: -114.2544403, 271.7889099, -116.4843903, 274.8233948, -389.0778198, 388.2733154
2: -96.9108582, 299.3471985, -98.8284225, 302.7738953, -399.6847534, 398.1755981
3: -101.6959457, 389.3615417, -103.5972824, 393.0469360, -494.7428894, 492.9588318
4: -86.8806610, 353.7731018, -88.4753418, 357.5619202, -444.4425659, 442.2484131

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7427272, upper bound: 339.7543219
time: 1.01 seconds

## Relational analysis of NS_B1_B1_A2_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7211445, upper bound: 339.7489066
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -83.0684814, 278.6816711, -81.4270630, 270.8995056, -353.9679871, 360.1087341
1: -116.6364975, 276.4191284, -114.1825333, 269.1073914, -385.7438660, 390.6016541
2: -98.9037476, 304.4310913, -96.8973541, 296.5032349, -395.4069824, 401.3284302
3: -103.7691269, 396.0045166, -101.5671158, 385.0579529, -488.8270874, 497.5716248
4: -88.6151886, 359.9463806, -86.7820969, 350.4340515, -439.0492554, 446.7284851

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_A1_A2_B1_B1

### Relational analysis result of NS_B1_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7891741, upper bound: 339.7925126
time: 0.88 seconds

## Relational analysis of NS_B1_B1_A2_A1_A2_B1_B2

### Relational analysis result of NS_B1_B1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889762, upper bound: 339.7886291
time: 0.84 seconds

## BFS NS instance: NS_B1_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -83.0684814, 278.6816711, -83.0612259, 276.6855164, -359.7539368, 361.7428589
1: -116.6364975, 276.4191284, -116.4843903, 274.8233948, -391.4598999, 392.9035034
2: -98.9037476, 304.4310913, -98.8284225, 302.7738953, -401.6776428, 403.2595215
3: -103.7691269, 396.0045166, -103.5972824, 393.0469360, -496.8160706, 499.6018066
4: -88.6151886, 359.9463806, -88.4753418, 357.5619202, -446.1770935, 448.4217224

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_B1_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7891741, upper bound: 339.7950727
time: 1.41 seconds

## Relational analysis of NS_B1_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_B1_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889762, upper bound: 339.7922973
time: 0.85 seconds

## BFS NS instance: NS_B1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -84.0325317, 282.1983337, -80.4826736, 267.5901794, -351.6227112, 362.6809998
1: -118.0285568, 279.9098511, -112.8496170, 265.7628174, -383.7913513, 392.7593384
2: -100.0767822, 308.2475281, -95.7479324, 292.7885742, -392.8653564, 403.9953918
3: -104.9913025, 400.8173523, -100.3842545, 380.3419189, -485.3332214, 501.2015686
4: -89.6422729, 364.2773132, -85.7559891, 346.1466980, -435.7889709, 450.0332947

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_B1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973746, upper bound: 339.7897315
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017501, upper bound: 339.7906850
time: 1.24 seconds

## BFS NS instance: NS_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -84.0325317, 282.1983337, -81.8182907, 272.2176514, -356.2501831, 364.0166321
1: -118.0285568, 279.9098511, -114.7416687, 270.3858032, -388.4143677, 394.6514587
2: -100.0767822, 308.2475281, -97.3347473, 297.8790894, -397.9558716, 405.5822449
3: -104.9913025, 400.8173523, -102.0502548, 386.7622681, -491.7535706, 502.8676147
4: -89.6422729, 364.2773132, -87.1439819, 351.9501038, -441.5923767, 451.4212952

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973746, upper bound: 339.8033265
time: 1.02 seconds

## Relational analysis of NS_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017501, upper bound: 339.8040532
time: 0.93 seconds

## BFS NS instance: NS_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -80.8200302, 268.8648682, -353.5058899, 365.1014404
1: -118.8840942, 281.9696045, -113.3218536, 267.1051025, -385.9891968, 395.2914429
2: -100.8020096, 310.5105286, -96.1628876, 294.3084106, -395.1104126, 406.6734009
3: -105.7517395, 403.7609863, -100.8058624, 382.2045288, -487.9562073, 504.5668335
4: -90.2919159, 366.9395752, -86.1291962, 347.8483276, -438.1402283, 453.0687866

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_B1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7634926, upper bound: 339.7771973
time: 0.77 seconds

## Relational analysis of NS_B1_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_B1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8036837, upper bound: 339.7990598
time: 0.73 seconds

## BFS NS instance: NS_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -82.4752197, 274.7119751, -359.3529968, 366.7566223
1: -118.8840942, 281.9696045, -115.6550064, 272.8845825, -391.7686768, 397.6245728
2: -100.8020096, 310.5105286, -98.1201096, 300.6493835, -401.4513855, 408.6306458
3: -105.7517395, 403.7609863, -102.8633804, 390.2746582, -496.0263367, 506.6243591
4: -90.2919159, 366.9395752, -87.8455429, 355.0695496, -445.3614502, 454.7851257

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_B1_B1_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7663290, upper bound: 339.7656788
time: 1.01 seconds

## Relational analysis of NS_B1_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_B1_B1_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7619737, upper bound: 339.7645070
time: 0.92 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -82.9631195, 276.8135071, -81.1084824, 271.5228882, -354.4860229, 357.9219971
1: -116.3883820, 274.7933960, -113.8824463, 269.4772644, -385.8656616, 388.6758423
2: -98.7218552, 302.6900635, -96.5852280, 296.8337708, -395.5556335, 399.2752991
3: -103.5415497, 393.3660583, -101.2940826, 385.7579956, -489.2995605, 494.6601562
4: -88.4102249, 357.7872925, -86.5218964, 350.7583008, -439.1685181, 444.3091736

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7984459, upper bound: 339.7949967
time: 1.20 seconds

## Relational analysis of NS_B1_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976989, upper bound: 339.7948735
time: 0.76 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -85.0805206, 283.3811340, -81.0518646, 271.3263855, -356.4069214, 364.4329834
1: -119.3118439, 281.4162292, -113.8008347, 269.2829590, -388.5947876, 395.2170715
2: -101.2236862, 309.9487915, -96.5163803, 296.6203003, -397.8439636, 406.4651489
3: -106.1273575, 402.6170959, -101.2217102, 385.4770813, -491.6044312, 503.8388062
4: -90.6223145, 366.1806335, -86.4604492, 350.5027771, -441.1250610, 452.6410828

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8018117, upper bound: 339.7960232
time: 0.94 seconds

## Relational analysis of NS_B1_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8014654, upper bound: 339.7956773
time: 0.95 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -82.0677032, 273.9620361, -81.6958694, 273.5408630, -355.6085815, 355.6578979
1: -115.1054993, 271.9100342, -114.7062836, 271.4855042, -386.5909729, 386.6162109
2: -97.6527023, 299.5496826, -97.2843628, 299.0462952, -396.6989441, 396.8340454
3: -102.4091339, 389.4417114, -102.0274506, 388.6209106, -491.0300293, 491.4691772
4: -87.4815598, 354.2571106, -87.1474457, 353.3419495, -440.8234558, 441.4045410

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7852993, upper bound: 339.7636294
time: 0.86 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8007421, upper bound: 339.7965661
time: 1.04 seconds

## BFS NS instance: NS_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -83.6558990, 279.2701721, -81.6958694, 273.5408630, -357.1966858, 360.9660339
1: -117.3529510, 277.2351685, -114.7062836, 271.4855042, -388.8384399, 391.9414062
2: -99.5460052, 305.3917542, -97.2843628, 299.0462952, -398.5922852, 402.6761169
3: -104.3943710, 396.8941956, -102.0274506, 388.6209106, -493.0152893, 498.9216309
4: -89.1501999, 360.9986877, -87.1474457, 353.3419495, -442.4921265, 448.1461182

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7852993, upper bound: 339.7636294
time: 1.47 seconds

## Relational analysis of NS_B1_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8007421, upper bound: 339.7965661
time: 1.10 seconds

## BFS NS instance: NS_B1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -83.5634842, 278.9434814, -81.0941696, 271.8367615, -355.4002380, 360.0376587
1: -117.2282486, 276.9187622, -113.8919601, 269.8121033, -387.0403442, 390.8106995
2: -99.4570312, 305.0360413, -96.6003113, 297.2022095, -396.6592407, 401.6363220
3: -104.2830505, 396.3751831, -101.2907791, 386.0637207, -490.3467712, 497.6659241
4: -89.0711365, 360.4910889, -86.5188522, 350.9210510, -439.9921875, 447.0099487

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651764, upper bound: 339.7665969
time: 0.93 seconds

## Relational analysis of NS_B1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651764, upper bound: 339.7665969
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -81.4670258, 274.0292664, -81.6958694, 273.5408630, -355.0078735, 355.7251282
1: -114.2544403, 271.7889099, -114.7062836, 271.4855042, -385.7399292, 386.4951782
2: -96.9108582, 299.3471985, -97.2843628, 299.0462952, -395.9571533, 396.6315613
3: -101.6959457, 389.3615417, -102.0274506, 388.6209106, -490.3168640, 491.3889771
4: -86.8806610, 353.7731018, -87.1474457, 353.3419495, -440.2225952, 440.9205322

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_B1_B2_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7216400, upper bound: 339.7394392
time: 1.00 seconds

## Relational analysis of NS_B1_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_B1_B2_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7191992, upper bound: 339.7195119
time: 1.04 seconds

## BFS NS instance: NS_B1_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -83.0684814, 278.6816711, -81.6958694, 273.5408630, -356.6093445, 360.3775330
1: -116.6364975, 276.4191284, -114.7062836, 271.4855042, -388.1219788, 391.1253052
2: -98.9037476, 304.4310913, -97.2843628, 299.0462952, -397.9500427, 401.7154541
3: -103.7691269, 396.0045166, -102.0274506, 388.6209106, -492.3900452, 498.0319824
4: -88.6151886, 359.9463806, -87.1474457, 353.3419495, -441.9571228, 447.0938110

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_B1_B2_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7216477, upper bound: 339.7378261
time: 0.92 seconds

## Relational analysis of NS_B1_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_B1_B2_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7193753, upper bound: 339.7194724
time: 0.80 seconds

## BFS NS instance: NS_B1_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -81.4670258, 274.0292664, -83.1899185, 278.9436035, -360.4105835, 357.2191772
1: -114.2544403, 271.7889099, -116.8321838, 276.8170471, -391.0714722, 388.6210938
2: -96.9108582, 299.3471985, -99.0710449, 304.8868408, -401.7976990, 398.4181519
3: -101.6959457, 389.3615417, -103.9003830, 396.2039795, -497.8999329, 493.2619324
4: -86.8806610, 353.7731018, -88.7171707, 360.1357422, -447.0164185, 442.4902649

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A2_A1_B2_A1_B1

### Relational analysis result of NS_B1_B2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7217294, upper bound: 339.7454907
time: 1.00 seconds

## Relational analysis of NS_B1_B2_A2_A1_B2_A1_B2

### Relational analysis result of NS_B1_B2_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7217192, upper bound: 339.7479330
time: 0.92 seconds

## BFS NS instance: NS_B1_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -83.0684814, 278.6816711, -83.1899185, 278.9436035, -362.0120544, 361.8715820
1: -116.6364975, 276.4191284, -116.8321838, 276.8170471, -393.4535522, 393.2513123
2: -98.9037476, 304.4310913, -99.0710449, 304.8868408, -403.7905884, 403.5020752
3: -103.7691269, 396.0045166, -103.9003830, 396.2039795, -499.9731140, 499.9049072
4: -88.6151886, 359.9463806, -88.7171707, 360.1357422, -448.7509155, 448.6635437

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B1_B2_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7217268, upper bound: 339.7433467
time: 1.11 seconds

## Relational analysis of NS_B1_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B1_B2_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7217159, upper bound: 339.7464965
time: 0.95 seconds

## BFS NS instance: NS_B1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -81.9757233, 275.0472107, -79.4468994, 265.6403198, -347.6160278, 354.4940491
1: -115.1538925, 272.8108826, -111.5879745, 263.6625671, -378.8164673, 384.3988647
2: -97.6510162, 300.4247742, -94.6373520, 290.3768616, -388.0278931, 395.0620728
3: -102.4304276, 390.5149841, -99.2310257, 377.1585999, -479.5890198, 489.7459717
4: -87.4778671, 354.9875793, -84.7496872, 343.0127563, -430.4906311, 439.7372131

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_B1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
time: 1.21 seconds

## Relational analysis of NS_B1_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_B1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
time: 1.02 seconds

## BFS NS instance: NS_B1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -81.4201355, 273.1122437, -86.6775894, 288.4697876, -369.8899231, 359.7898254
1: -114.2983780, 270.9042053, -121.7378235, 286.4046936, -400.7030640, 392.6420288
2: -96.9229050, 298.3837280, -103.2345963, 315.5512085, -412.4740295, 401.6183167
3: -101.6785431, 387.8798828, -108.1949463, 409.3881226, -511.0666504, 496.0748291
4: -86.8289490, 352.6365051, -92.2977448, 372.7454834, -459.5744324, 444.9342651

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
time: 0.90 seconds

## Relational analysis of NS_B1_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
time: 1.23 seconds

## BFS NS instance: NS_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -81.1008835, 271.5277100, -356.1687927, 365.3822632
1: -118.8840942, 281.9696045, -113.8579330, 269.5082397, -388.3923340, 395.8274841
2: -100.8020096, 310.5105286, -96.5607834, 296.8796692, -397.6816711, 407.0713196
3: -105.7517395, 403.7609863, -101.2776871, 385.7951660, -491.5468445, 505.0386658
4: -90.2919159, 366.9395752, -86.5050583, 350.7838135, -441.0757446, 453.4446411

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_B2_A2_A2_B2_B1_B1

### Relational analysis result of NS_B1_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7833407, upper bound: 339.7616305
time: 1.00 seconds

## Relational analysis of NS_B1_B2_A2_A2_B2_B1_B2

### Relational analysis result of NS_B1_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017064, upper bound: 339.7957794
time: 0.98 seconds

## BFS NS instance: NS_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -82.6239700, 277.0314331, -361.6724548, 366.9053650
1: -118.8840942, 281.9696045, -116.0265427, 274.9391174, -393.8231812, 397.9961548
2: -100.8020096, 310.5105286, -98.3831253, 302.8294983, -403.6315002, 408.8936462
3: -105.7517395, 403.7609863, -103.1880188, 393.5193481, -499.2710571, 506.9490051
4: -90.2919159, 366.9395752, -88.1063156, 357.7086792, -448.0005798, 455.0458984

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B2_A2_A2_B2_B2_B1

### Relational analysis result of NS_B1_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7628488, upper bound: 339.7799483
time: 1.01 seconds

## Relational analysis of NS_B1_B2_A2_A2_B2_B2_B2

### Relational analysis result of NS_B1_B2_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7583736, upper bound: 339.7668440
time: 1.04 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -80.5952835, 269.7451477, -352.4299622, 356.6342468
1: -115.9796906, 273.9595032, -112.9131546, 267.7285461, -383.7082520, 386.8726501
2: -98.3984375, 301.7886658, -95.8295746, 294.9547119, -393.3531189, 397.6182251
3: -103.1827164, 392.3468323, -100.5038986, 383.3568726, -486.5395813, 492.8507385
4: -88.1445007, 356.8894043, -85.8979874, 348.5716553, -436.7161560, 442.7873840

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7326569, upper bound: 339.7497506
time: 1.06 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7301529, upper bound: 339.7278024
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -82.1501541, 274.2453003, -356.9301147, 358.1891479
1: -115.9796906, 273.9595032, -115.2254028, 272.1891479, -388.1688232, 389.1849060
2: -98.3984375, 301.7886658, -97.7641296, 299.8466797, -398.2451172, 399.5527954
3: -103.1827164, 392.3468323, -102.5127182, 389.8078003, -492.9905090, 494.8595581
4: -88.1445007, 356.8894043, -87.5782852, 354.5833130, -442.7278137, 444.4676819

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_B2_A1_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7330293, upper bound: 339.7584513
time: 1.19 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_B2_A1_B1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7300900, upper bound: 339.7300900
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -82.6270142, 276.5968323, -82.3742294, 274.9334106, -357.5604248, 358.9710083
1: -115.7941513, 274.5624695, -115.5383682, 272.8794861, -388.6736145, 390.1008301
2: -98.2563629, 302.4519043, -98.0269623, 300.6091309, -398.8654785, 400.4788208
3: -103.0518112, 393.0264587, -102.7902451, 390.7839355, -493.8357544, 495.8167114
4: -88.0448837, 357.2890930, -87.8121414, 355.4825745, -443.5274658, 445.1011658

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7520649, upper bound: 339.7661487
time: 1.34 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7451847, upper bound: 339.7311004
time: 1.21 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -83.9726944, 280.4361267, -82.3742294, 274.9334106, -358.9060974, 362.8103638
1: -117.8029175, 278.3698730, -115.5383682, 272.8794861, -390.6823730, 393.9082336
2: -99.9330673, 306.6219177, -98.0269623, 300.6091309, -400.5422058, 404.6488342
3: -104.7941132, 398.5148315, -102.7902451, 390.7839355, -495.5780640, 501.3050537
4: -89.4950256, 362.4327087, -87.8121414, 355.4825745, -444.9775391, 450.2447815

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7619852, upper bound: 339.7666167
time: 1.13 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7538543, upper bound: 339.7314868
time: 1.24 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -83.5634842, 278.9434814, -81.7682877, 272.8135986, -356.3770752, 360.7117310
1: -117.2282486, 276.9187622, -114.6960449, 270.8885193, -388.1167603, 391.6147766
2: -99.4570312, 305.0360413, -97.3384552, 298.4335938, -397.8906250, 402.3745117
3: -104.2830505, 396.3751831, -102.0365982, 387.6362610, -491.9193115, 498.4117737
4: -89.0711365, 360.4910889, -87.1854553, 352.5321960, -441.6033325, 447.6765442

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7649082, upper bound: 339.7645402
time: 1.07 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7649082, upper bound: 339.7645402
time: 1.15 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.4556885, 278.5311279, -80.9158249, 271.2387695, -354.6944275, 359.4469604
1: -117.0780792, 276.4986572, -113.5990219, 269.1152344, -386.1932983, 390.0975952
2: -99.3010635, 304.5668945, -96.3396759, 296.3959351, -395.6969910, 400.9065552
3: -104.1580658, 395.8431702, -101.0660553, 385.4822998, -489.6403503, 496.9091797
4: -88.9326401, 360.0378113, -86.3224792, 350.4093628, -439.3420105, 446.3602600

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -82.7230530, 275.8771057, -81.6985779, 273.8495483, -356.5726013, 357.5756531
1: -116.0421448, 273.9121094, -114.6844330, 271.7105103, -387.7526550, 388.5965271
2: -98.4337540, 301.7283936, -97.2630920, 299.2796936, -397.7134399, 398.9914856
3: -103.2382660, 392.0785217, -102.0450058, 389.2391663, -492.4774170, 494.1235352
4: -88.1647110, 356.6487427, -87.1653900, 353.8341064, -441.9988098, 443.8141174

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -82.0677032, 273.9620361, -82.9877167, 278.3031006, -360.3707886, 356.9497681
1: -115.1054993, 271.9100342, -116.5166245, 276.0668640, -391.1723633, 388.4266052
2: -97.6527023, 299.5496826, -98.8013992, 304.0500183, -401.7027283, 398.3510742
3: -102.4091339, 389.4417114, -103.6618652, 395.4787903, -497.8879395, 493.1035461
4: -87.4815598, 354.2571106, -88.5221558, 359.4860229, -446.9675293, 442.7792664

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7838317, upper bound: 339.7629913
time: 1.07 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8004945, upper bound: 339.7969904
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -83.6558990, 279.2701721, -82.9877167, 278.3031006, -361.9589233, 362.2578735
1: -117.3529510, 277.2351685, -116.5166245, 276.0668640, -393.4197998, 393.7518005
2: -99.5460052, 305.3917542, -98.8013992, 304.0500183, -403.5960083, 404.1931458
3: -104.3943710, 396.8941956, -103.6618652, 395.4787903, -499.8731689, 500.5560303
4: -89.1501999, 360.9986877, -88.5221558, 359.4860229, -448.6362000, 449.5208435

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7838317, upper bound: 339.7629913
time: 1.05 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8004945, upper bound: 339.7969904
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -83.1922531, 278.0795288, -82.4355927, 276.8254089, -360.0176697, 360.5150452
1: -116.4557724, 276.1462402, -115.7892151, 274.5837402, -391.0395203, 391.9354553
2: -98.8304291, 304.3158875, -98.1944275, 302.4031677, -401.2335510, 402.5103149
3: -103.6560516, 395.5121155, -103.0001907, 393.1347046, -496.7907715, 498.5122986
4: -88.6291504, 359.5879211, -87.9582291, 357.2901611, -445.9193115, 447.5461426

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967455, upper bound: 339.8010276
time: 1.00 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911435, upper bound: 339.7937102
time: 1.23 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911435, upper bound: 339.8010276
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -83.1912079, 277.7921143, -83.6860199, 280.9848938, -364.1760864, 361.4781189
1: -116.7025604, 275.7407532, -117.5372162, 278.7214661, -395.4240112, 393.2779541
2: -98.9735184, 303.7406921, -99.6626129, 306.9438171, -405.9173279, 403.4033203
3: -103.8308334, 394.7933350, -104.5541153, 399.1029968, -502.9338379, 499.3474426
4: -88.6453552, 359.0614014, -89.2721252, 362.7269897, -451.3722534, 448.3335266

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8030964, upper bound: 339.8013813
time: 1.07 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8035764, upper bound: 339.8035671
time: 1.19 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -82.0677032, 273.9620361, -84.2925873, 283.0604248, -365.1281128, 358.2546387
1: -115.1054993, 271.9100342, -118.3896637, 280.7737732, -395.8792419, 390.2996521
2: -97.6527023, 299.5496826, -100.3853149, 309.1988220, -406.8514709, 399.9349976
3: -102.4091339, 389.4417114, -105.3119278, 402.0356445, -504.4447632, 494.7535706
4: -87.4815598, 354.2571106, -89.9195251, 365.3794861, -452.8610229, 444.1766357

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7838317, upper bound: 339.7640854
time: 0.89 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8004945, upper bound: 339.8035932
time: 1.05 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -83.6558990, 279.2701721, -84.2925873, 283.0604248, -366.7162476, 363.5627441
1: -117.3529510, 277.2351685, -118.3896637, 280.7737732, -398.1267090, 395.6248169
2: -99.5460052, 305.3917542, -100.3853149, 309.1988220, -408.7448120, 405.7770691
3: -104.3943710, 396.8941956, -105.3119278, 402.0356445, -506.4300232, 502.2060852
4: -89.1501999, 360.9986877, -89.9195251, 365.3794861, -454.5296936, 450.9182129

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7358195, upper bound: 339.7793113
time: 1.02 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7967321, upper bound: 339.7953688
time: 0.97 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7940867, upper bound: 339.7949001
time: 1.10 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -82.0553513, 275.2927246, -81.7826614, 273.1503601, -355.2056885, 357.0753784
1: -115.2062912, 273.0572205, -114.7188339, 271.1412964, -386.3475952, 387.7760620
2: -97.6846466, 300.7300110, -97.2911377, 298.6815491, -396.3662109, 398.0211487
3: -102.5005569, 391.1980286, -102.0702209, 388.2352295, -490.7357788, 493.2682495
4: -87.5280914, 355.5879822, -87.1461639, 353.0996399, -440.6277466, 442.7341309

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937009, upper bound: 339.7991011
time: 0.82 seconds

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## BFS NS instance: NS_B2_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -81.8563385, 274.5798035, -81.2660065, 271.2685242, -353.1248169, 355.8458252
1: -114.9373932, 272.3360596, -114.0043488, 269.2507019, -384.1881104, 386.3403931
2: -97.4592743, 299.9335022, -96.6996231, 296.5885315, -394.0477295, 396.6331177
3: -102.2579803, 390.1575928, -101.4280167, 385.5017090, -487.7596741, 491.5855713
4: -87.3258057, 354.6647339, -86.6178360, 350.6503601, -437.9761658, 441.2825623

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_B2_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7939909, upper bound: 339.7990870
time: 1.22 seconds

## Relational analysis of NS_B2_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_B2_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7938668, upper bound: 339.7984623
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -81.7641449, 272.8780823, -356.1644287, 361.1171265
1: -116.9414825, 277.0936890, -114.6737595, 270.8517761, -387.7932739, 391.7674561
2: -99.1595917, 305.1754456, -97.2892609, 298.3940430, -397.5535583, 402.4647217
3: -104.0396271, 396.9595032, -102.0252457, 387.9102173, -491.9498291, 498.9847412
4: -88.8424301, 360.8237305, -87.1565399, 352.8792725, -441.7217102, 447.9802551

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7629913, upper bound: 339.7838317
time: 0.81 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7969904, upper bound: 339.8004945
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -83.3108444, 278.0466919, -361.3330383, 362.6638489
1: -116.9414825, 277.0936890, -116.8636551, 276.0385132, -392.9799805, 393.9573364
2: -99.1595917, 305.1754456, -99.1347122, 304.0859070, -403.2454529, 404.3100891
3: -104.0396271, 396.9595032, -103.9588852, 395.1679382, -499.2075500, 500.9183960
4: -88.8424301, 360.8237305, -88.7820511, 359.4390869, -448.2815247, 449.6057434

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_B2_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7629913, upper bound: 339.7842104
time: 0.96 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_B2_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7969904, upper bound: 339.8019336
time: 0.92 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -82.9927597, 279.4716492, -82.3742294, 274.9334106, -357.9261780, 361.8458557
1: -116.4448547, 277.1882324, -115.5383682, 272.8794861, -389.3243408, 392.7265930
2: -98.7608032, 305.2538147, -98.0269623, 300.6091309, -399.3698730, 403.2807312
3: -103.6259918, 396.9041748, -102.7902451, 390.7839355, -494.4099121, 499.6943970
4: -88.5176544, 360.5835571, -87.8121414, 355.4825745, -444.0002441, 448.3956604

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_B2_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7455900, upper bound: 339.7302575
time: 1.08 seconds

## Relational analysis of NS_B2_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_B2_A2_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7430198, upper bound: 339.7300586
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -84.3756714, 283.4887085, -82.3742294, 274.9334106, -359.3090515, 365.8629150
1: -118.5069656, 281.1713562, -115.5383682, 272.8794861, -391.3864136, 396.7097168
2: -100.4826660, 309.6351929, -98.0269623, 300.6091309, -401.0917664, 407.6621399
3: -105.4197922, 402.6337891, -102.7902451, 390.7839355, -496.2037048, 505.4240112
4: -90.0084457, 365.9017029, -87.8121414, 355.4825745, -445.4910278, 453.7137756

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7549596, upper bound: 339.7306764
time: 1.03 seconds

## Relational analysis of NS_B2_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7524395, upper bound: 339.7305998
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -81.1684723, 272.9712830, -356.2576294, 360.5214844
1: -116.9414825, 277.0936890, -113.8285828, 270.7551880, -387.6966553, 390.9222717
2: -99.1595917, 305.1754456, -96.5521164, 298.2187500, -397.3782959, 401.7275696
3: -104.0396271, 396.9595032, -101.3170624, 387.8692932, -491.9089050, 498.2765503
4: -88.8424301, 360.8237305, -86.5601654, 352.4267273, -441.2691650, 447.3839111

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7511819, upper bound: 339.7218285
time: 1.13 seconds

## Relational analysis of NS_B2_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7193032, upper bound: 339.7190950
time: 0.92 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -83.2863541, 279.3529968, -82.7713394, 277.6364136, -360.9227600, 362.1243286
1: -116.9414825, 277.0936890, -116.2136459, 275.3968811, -392.3383789, 393.3073425
2: -99.1595917, 305.1754456, -98.5472183, 303.3109131, -402.4704590, 403.7226257
3: -104.0396271, 396.9595032, -103.3931732, 394.5304871, -498.5700989, 500.3526306
4: -88.8424301, 360.8237305, -88.2964630, 358.6152039, -447.4576416, 449.1201782

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B1_A1_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7218011, upper bound: 339.7504787
time: 0.85 seconds

## Relational analysis of NS_B2_A2_B2_B1_A1_B2_B2

### Relational analysis result of NS_B2_A2_B2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7192741, upper bound: 339.7192741
time: 1.17 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -81.1684723, 272.9712830, -357.6123047, 365.4498901
1: -118.8840942, 281.9696045, -113.8285828, 270.7551880, -389.6392517, 395.7981567
2: -100.8020096, 310.5105286, -96.5521164, 298.2187500, -399.0207520, 407.0626526
3: -105.7517395, 403.7609863, -101.3170624, 387.8692932, -493.6210022, 505.0780640
4: -90.2919159, 366.9395752, -86.5601654, 352.4267273, -442.7186279, 453.4997559

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7532158, upper bound: 339.7219664
time: 0.93 seconds

## Relational analysis of NS_B2_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7502781, upper bound: 339.7218219
time: 1.12 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -84.6411057, 284.2814026, -82.7713394, 277.6364136, -362.2773743, 367.0527344
1: -118.8840942, 281.9696045, -116.2136459, 275.3968811, -394.2809753, 398.1832275
2: -100.8020096, 310.5105286, -98.5472183, 303.3109131, -404.1129150, 409.0577393
3: -105.7517395, 403.7609863, -103.3931732, 394.5304871, -500.2821655, 507.1541443
4: -90.2919159, 366.9395752, -88.2964630, 358.6152039, -448.9071045, 455.2360229

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7560925, upper bound: 339.7582822
time: 1.05 seconds

## Relational analysis of NS_B2_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7493160, upper bound: 339.7218327
time: 1.11 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -83.1754227, 279.0008240, -81.8092346, 274.3444519, -357.5198669, 360.8100586
1: -116.8103027, 276.8113098, -114.8410645, 272.5559082, -389.3661499, 391.6523438
2: -99.0491333, 304.8506775, -97.3930130, 300.3125000, -399.3616333, 402.2436829
3: -103.9177628, 396.3858643, -102.1685486, 390.4936218, -494.4113770, 498.5544128
4: -88.7230453, 360.3223267, -87.2895432, 355.1175537, -443.8405762, 447.6118774

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
time: 1.06 seconds

## Relational analysis of NS_B2_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -82.8717575, 278.0109863, -80.7232819, 270.9263306, -353.7980957, 358.7342529
1: -116.3453903, 275.8290100, -113.2673721, 268.8335876, -385.1789856, 389.0963745
2: -98.6523819, 303.7762756, -96.0656662, 296.1170959, -394.7694702, 399.8419189
3: -103.5135651, 395.0334473, -100.7871017, 384.9802856, -488.4938354, 495.8205566
4: -88.3739624, 359.0885315, -86.0868912, 349.8993835, -438.2733459, 445.1753845

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -82.6941681, 277.3472900, -84.2925873, 283.0604248, -365.7545471, 361.6398315
1: -116.1006012, 275.1199341, -118.3896637, 280.7737732, -396.8743591, 393.5095825
2: -98.4422455, 303.0119629, -100.3853149, 309.1988220, -407.6410522, 403.3972778
3: -103.2963638, 394.1492310, -105.3119278, 402.0356445, -505.3319702, 499.4611206
4: -88.2055588, 358.2816772, -89.9195251, 365.3794861, -453.5850525, 448.2012024

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7610205, upper bound: 339.7819196
time: 1.02 seconds

## Relational analysis of NS_B2_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962459, upper bound: 339.8013895
time: 1.02 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -84.0756302, 282.3734131, -84.2925873, 283.0604248, -367.1360474, 366.6659851
1: -118.0820465, 280.0888977, -118.3896637, 280.7737732, -398.8557739, 398.4785767
2: -100.1168289, 308.4468994, -100.3853149, 309.1988220, -409.3156433, 408.8322144
3: -105.0422668, 401.0846863, -105.3119278, 402.0356445, -507.0779114, 506.3966064
4: -89.6833267, 364.5166626, -89.9195251, 365.3794861, -455.0628052, 454.4361877

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7281915, upper bound: 339.7828932
time: 0.91 seconds

## Relational analysis of NS_B2_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_B2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7258080, upper bound: 339.7668443
time: 0.89 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.76 seconds
NS_B1_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7985034, upper bound: 339.7895439
NS_B1_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8000652, upper bound: 339.7904316
NS_B1_B1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7907824, upper bound: 339.7695433
NS_B1_B1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8027122, upper bound: 339.8003615
NS_B1_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7996897, upper bound: 339.8007935
NS_B1_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8019150, upper bound: 339.8036706
NS_B1_B1_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7376770, upper bound: 339.7548072
NS_B1_B1_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7371150, upper bound: 339.7547895
NS_B1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7983758, upper bound: 339.7894986
NS_B1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8017527, upper bound: 339.7904214
NS_B1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7710511, upper bound: 339.7811421
NS_B1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8042069, upper bound: 339.8002647
NS_B1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7985200, upper bound: 339.8004114
NS_B1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8038537, upper bound: 339.8038996
NS_B1_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7672659, upper bound: 339.7757611
NS_B1_B1_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7647153, upper bound: 339.7643158
NS_B1_B1_A2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7427272, upper bound: 339.7305326
NS_B1_B1_A2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7211445, upper bound: 339.7281173
NS_B1_B1_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7427272, upper bound: 339.7543219
NS_B1_B1_A2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7211445, upper bound: 339.7489066
NS_B1_B1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7891741, upper bound: 339.7925126
NS_B1_B1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7889762, upper bound: 339.7886291
NS_B1_B1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7891741, upper bound: 339.7950727
NS_B1_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7889762, upper bound: 339.7922973
NS_B1_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7973746, upper bound: 339.7897315
NS_B1_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8017501, upper bound: 339.7906850
NS_B1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7973746, upper bound: 339.8033265
NS_B1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8017501, upper bound: 339.8040532
NS_B1_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7634926, upper bound: 339.7771973
NS_B1_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8036837, upper bound: 339.7990598
NS_B1_B1_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7663290, upper bound: 339.7656788
NS_B1_B1_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7619737, upper bound: 339.7645070
NS_B1_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7984459, upper bound: 339.7949967
NS_B1_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7976989, upper bound: 339.7948735
NS_B1_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8018117, upper bound: 339.7960232
NS_B1_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8014654, upper bound: 339.7956773
NS_B1_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7852993, upper bound: 339.7636294
NS_B1_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8007421, upper bound: 339.7965661
NS_B1_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7852993, upper bound: 339.7636294
NS_B1_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8007421, upper bound: 339.7965661
NS_B1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7651764, upper bound: 339.7665969
NS_B1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7651764, upper bound: 339.7665969
NS_B1_B2_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7216400, upper bound: 339.7394392
NS_B1_B2_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7191992, upper bound: 339.7195119
NS_B1_B2_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7216477, upper bound: 339.7378261
NS_B1_B2_A2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7193753, upper bound: 339.7194724
NS_B1_B2_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7217294, upper bound: 339.7454907
NS_B1_B2_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7217192, upper bound: 339.7479330
NS_B1_B2_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7217268, upper bound: 339.7433467
NS_B1_B2_A2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7217159, upper bound: 339.7464965
NS_B1_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
NS_B1_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
NS_B1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
NS_B1_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7994139, upper bound: 339.7996731
NS_B1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7833407, upper bound: 339.7616305
NS_B1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8017064, upper bound: 339.7957794
NS_B1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7628488, upper bound: 339.7799483
NS_B1_B2_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7583736, upper bound: 339.7668440
NS_B2_A1_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7326569, upper bound: 339.7497506
NS_B2_A1_B1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7301529, upper bound: 339.7278024
NS_B2_A1_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7330293, upper bound: 339.7584513
NS_B2_A1_B1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7300900, upper bound: 339.7300900
NS_B2_A1_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7520649, upper bound: 339.7661487
NS_B2_A1_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7451847, upper bound: 339.7311004
NS_B2_A1_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7619852, upper bound: 339.7666167
NS_B2_A1_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7538543, upper bound: 339.7314868
NS_B2_A1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7649082, upper bound: 339.7645402
NS_B2_A1_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7649082, upper bound: 339.7645402
NS_B2_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7838317, upper bound: 339.7629913
NS_B2_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8004945, upper bound: 339.7969904
NS_B2_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7838317, upper bound: 339.7629913
NS_B2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8004945, upper bound: 339.7969904
NS_B2_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7911435, upper bound: 339.7937102
NS_B2_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7911435, upper bound: 339.8010276
NS_B2_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8030964, upper bound: 339.8013813
NS_B2_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8035764, upper bound: 339.8035671
NS_B2_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7838317, upper bound: 339.7640854
NS_B2_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8004945, upper bound: 339.8035932
NS_B2_A1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7967321, upper bound: 339.7953688
NS_B2_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7940867, upper bound: 339.7949001
NS_B2_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7939909, upper bound: 339.7990870
NS_B2_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7938668, upper bound: 339.7984623
NS_B2_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7629913, upper bound: 339.7838317
NS_B2_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7969904, upper bound: 339.8004945
NS_B2_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7629913, upper bound: 339.7842104
NS_B2_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7969904, upper bound: 339.8019336
NS_B2_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7455900, upper bound: 339.7302575
NS_B2_A2_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7430198, upper bound: 339.7300586
NS_B2_A2_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7549596, upper bound: 339.7306764
NS_B2_A2_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7524395, upper bound: 339.7305998
NS_B2_A2_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7511819, upper bound: 339.7218285
NS_B2_A2_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7193032, upper bound: 339.7190950
NS_B2_A2_B2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7218011, upper bound: 339.7504787
NS_B2_A2_B2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7192741, upper bound: 339.7192741
NS_B2_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7532158, upper bound: 339.7219664
NS_B2_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7502781, upper bound: 339.7218219
NS_B2_A2_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7560925, upper bound: 339.7582822
NS_B2_A2_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7493160, upper bound: 339.7218327
NS_B2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
NS_B2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
NS_B2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
NS_B2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.8013772, upper bound: 339.8013764
NS_B2_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7610205, upper bound: 339.7819196
NS_B2_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7962459, upper bound: 339.8013895
NS_B2_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7281915, upper bound: 339.7828932
NS_B2_A2_B2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -339.7258080, upper bound: 339.7668443

## BFS NS instance: NS_B1_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -82.4448929, 275.6384888, -79.3230972, 263.7017517, -346.1466064, 354.9615173
1: -115.3451462, 273.7505798, -111.2300644, 261.9135742, -377.2587280, 384.9806519
2: -97.9420624, 301.7023926, -94.3932037, 288.5660095, -386.5080261, 396.0955811
3: -102.6775742, 392.1083069, -98.9439240, 374.7892456, -477.4667969, 491.0522156
4: -87.9024048, 356.5728455, -84.5421906, 341.0900879, -428.9924927, 441.1150513

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985034, upper bound: 339.7895439
time: 1.05 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7950082, upper bound: 339.7873583
time: 0.68 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924961, upper bound: 339.7888635
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924961, upper bound: 339.7895439
time: 0.92 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -81.4815598, 272.0960693, -80.4826736, 267.5901794, -349.0717163, 352.5787354
1: -114.2825546, 270.0409546, -112.8496170, 265.7628174, -380.0453796, 382.8904724
2: -96.9512024, 297.4815369, -95.7479324, 292.7885742, -389.7397156, 393.2293701
3: -101.6803207, 386.7278748, -100.3842545, 380.3419189, -482.0222473, 487.1120911
4: -86.8566132, 351.7770691, -85.7559891, 346.1466980, -433.0032959, 437.5330505

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7974685, upper bound: 339.7880362
time: 0.93 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8000652, upper bound: 339.7904316
time: 1.27 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985034, upper bound: 339.7902169
time: 0.96 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -78.7586060, 262.9025879, -345.5874023, 354.7976074
1: -115.9796906, 273.9595032, -110.2723160, 261.1591187, -377.1387939, 384.2318115
2: -98.3984375, 301.7886658, -93.6015625, 287.7864685, -386.1849060, 395.3901978
3: -103.1827164, 392.3468323, -98.1530838, 373.6403198, -476.8230286, 490.4999084
4: -88.1445007, 356.8894043, -83.8925552, 339.8811951, -428.0256958, 440.7819519

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A1_B1_B2_B1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7316396, upper bound: 339.7431683
time: 0.86 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B2_B1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7309077, upper bound: 339.7324113
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -82.6848221, 276.0390015, -80.6413803, 268.3112793, -350.9960938, 356.6803894
1: -115.9796906, 273.9595032, -113.0720215, 266.5502625, -382.5299683, 387.0315247
2: -98.3984375, 301.7886658, -95.9518356, 293.6934814, -392.0919189, 397.7405090
3: -103.1827164, 392.3468323, -100.5839996, 381.4231873, -484.6058960, 492.9307861
4: -88.1445007, 356.8894043, -85.9418716, 347.1309814, -435.2754822, 442.8312683

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A1_B1_B2_B2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7316450, upper bound: 339.7480133
time: 1.05 seconds

## Relational analysis of NS_B1_B1_A1_A1_B1_B2_B2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7304599, upper bound: 339.7297766
time: 1.22 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -82.4448929, 275.6384888, -80.5688629, 268.0504761, -350.4953613, 356.2073364
1: -115.3451462, 273.7505798, -112.9893646, 266.2518311, -381.5969849, 386.7399292
2: -97.9420624, 301.7023926, -95.8671570, 293.3335266, -391.2755432, 397.5695190
3: -102.6775742, 392.1083069, -100.4920425, 380.8001099, -483.4776306, 492.6003418
4: -87.9024048, 356.5728455, -85.8287888, 346.5080872, -434.4104919, 442.4016113

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962863, upper bound: 339.7984918
time: 0.80 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7877136, upper bound: 339.7900596
time: 0.83 seconds

## BFS NS instance: NS_B1_B1_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -81.4815598, 272.0960693, -81.8182907, 272.2176514, -353.6991882, 353.9143677
1: -114.2825546, 270.0409546, -114.7416687, 270.3858032, -384.6683655, 384.7825928
2: -96.9512024, 297.4815369, -97.3347473, 297.8790894, -394.8302612, 394.8162537
3: -101.6803207, 386.7278748, -102.0502548, 386.7622681, -488.4425964, 488.7781372
4: -86.8566132, 351.7770691, -87.1439819, 351.9501038, -438.8067017, 438.9210510

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7983334, upper bound: 339.8006293
time: 0.92 seconds

## Relational analysis of NS_B1_B1_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_B1_B1_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7989043, upper bound: 339.8036341
time: 1.45 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -83.6338425, 279.5299072, -79.3230972, 263.7017517, -347.3355408, 358.8529663
1: -117.0231857, 277.7009888, -111.2300644, 261.9135742, -378.9367676, 388.9310608
2: -99.3513641, 306.0446777, -94.3932037, 288.5660095, -387.9173584, 400.4378357
3: -104.1550751, 397.6117249, -98.9439240, 374.7892456, -478.9443054, 496.5556641
4: -89.1374283, 361.6022949, -84.5421906, 341.0900879, -430.2274475, 446.1444702

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7983758, upper bound: 339.7894986
time: 0.98 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962387, upper bound: 339.7890991
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962387, upper bound: 339.7894986
time: 0.81 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -82.9343796, 276.9708862, -80.4826736, 267.5901794, -350.5245667, 357.4535522
1: -116.3383636, 274.9323425, -112.8496170, 265.7628174, -382.1011963, 387.7818604
2: -98.6842041, 302.8461914, -95.7479324, 292.7885742, -391.4727173, 398.5940552
3: -103.4963455, 393.6072998, -100.3842545, 380.3419189, -483.8382263, 493.9915466
4: -88.3830795, 357.9421082, -85.7559891, 346.1466980, -434.5297852, 443.6980896

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017527, upper bound: 339.7902372
time: 0.72 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8017527, upper bound: 339.7904214
time: 1.35 seconds

## BFS NS instance: NS_B1_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -82.6270142, 276.5968323, -80.8200302, 268.8648682, -351.4918823, 357.4168701
1: -115.7941513, 274.5624695, -113.3218536, 267.1051025, -382.8992310, 387.8843384
2: -98.2563629, 302.4519043, -96.1628876, 294.3084106, -392.5647583, 398.6148071
3: -103.0518112, 393.0264587, -100.8058624, 382.2045288, -485.2563477, 493.8323364
4: -88.0448837, 357.2890930, -86.1291962, 347.8483276, -435.8932190, 443.4182739

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7545549, upper bound: 339.7650276
time: 1.01 seconds

## Relational analysis of NS_B1_B1_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_B1_A1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7468315, upper bound: 339.7313022
time: 0.95 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.97 + 418.39 = 421.36 seconds
