## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 398.85261092052


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315)
1: (-197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482)
2: (-197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371)
3: (-234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619)
4: (-201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.67 + 2.56 = 3.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9323974, upper bound: 398.9323973

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9180450, upper bound: 398.8849160
time: 1.10 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8848071
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.01 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 0, lower bound: -398.9180450, upper bound: 398.8849160
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8848071

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -179.2202911, 303.3570251, -468.2398987, 452.8569946
1: -181.5148926, 242.5971832, -197.4927673, 268.0154114, -449.5303040, 440.0898438
2: -181.4661713, 246.9006042, -197.7517548, 272.0600586, -453.5261230, 444.6523438
3: -214.6846161, 279.4453430, -234.1109924, 308.6250000, -523.3095093, 513.5563354
4: -184.8080750, 283.3780212, -201.8509827, 312.5909424, -497.3990173, 485.2290039

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8845823
time: 1.10 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8845823
time: 0.78 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -382.0293579, 576.7352295, -178.0285950, 301.3497620, -670.6077271, 750.0909424
1: -417.7243652, 527.1749878, -196.1907959, 266.3074036, -672.2918701, 718.0180664
2: -416.6148071, 536.4212646, -196.4221954, 270.3563843, -677.1156006, 727.4710693
3: -487.5961609, 608.3463135, -232.5894928, 306.6549072, -785.5539551, 834.6773071
4: -418.1431580, 614.5153198, -200.5400543, 310.5973816, -720.4613037, 812.7661743

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071
time: 0.85 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8845823
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8845823
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9151972, upper bound: 398.8835894
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8925887, upper bound: 398.8833715
time: 0.90 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -381.9187012, 576.5624390, -736.9779053, 642.8848877
1: -181.5148926, 242.5971832, -417.5991211, 527.0137329, -703.3931274, 648.6752319
2: -181.4661713, 246.9006042, -416.4932556, 536.2599487, -712.5161743, 654.1002197
3: -214.6846161, 279.4453430, -487.4469604, 608.1581421, -816.7816162, 758.2739868
4: -184.8080750, 283.3780212, -418.0183105, 614.3297119, -796.9906616, 693.2182007

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9151972, upper bound: 398.8835894
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8925887, upper bound: 398.8833715
time: 0.95 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -381.9187012, 576.5624390, -164.8828735, 273.6366882, -642.8849487, 736.9779053
1: -417.5991211, 527.0137329, -181.5148926, 242.5971832, -648.6752319, 703.3931274
2: -416.4932556, 536.2599487, -181.4661713, 246.9006042, -654.1002197, 712.5161743
3: -487.4469604, 608.1581421, -214.6846161, 279.4453430, -758.2739868, 816.7816162
4: -418.0183105, 614.3297119, -184.8080750, 283.3780212, -693.2182007, 796.9906616

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828524, upper bound: 398.8828584
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832703, upper bound: 398.8835811
time: 1.05 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828524, upper bound: 398.8828584
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832703, upper bound: 398.8835811
time: 0.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.28 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.9151972, upper bound: 398.8835894
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.8925887, upper bound: 398.8833715
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.9151972, upper bound: 398.8835894
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.8925887, upper bound: 398.8833715
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.8828524, upper bound: 398.8828584
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.8832703, upper bound: 398.8835811
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.8828524, upper bound: 398.8828584
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -398.8832703, upper bound: 398.8835811

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -164.8828735, 273.6366882, -434.5151062, 430.8043823
1: -177.0686951, 235.9393463, -181.5148926, 242.5971832, -419.6657104, 417.4541626
2: -176.9501801, 240.2561340, -181.4661713, 246.9006042, -423.8507690, 421.7222900
3: -209.3154907, 271.8285217, -214.6846161, 279.4453430, -488.7608337, 486.5131226
4: -180.1105499, 275.6749573, -184.8080750, 283.3780212, -463.4885864, 460.4830322

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -182.5338745, 294.6768799, -163.1987305, 270.5172119, -453.0510559, 457.8756104
1: -200.3133240, 262.0391235, -179.6420898, 239.8729553, -440.1862793, 441.6811829
2: -200.4873199, 266.9307861, -179.5790405, 244.1992493, -444.6865845, 446.5097656
3: -235.6646729, 302.0430603, -212.4424133, 276.3217468, -511.9864197, 514.4854736
4: -202.8659515, 306.8235168, -182.8298950, 280.2348328, -483.1007690, 489.6533508

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -381.8544006, 576.4777832, -732.9840698, 635.0592651
1: -177.0686951, 235.9393463, -417.5255432, 526.9307861, -698.9297485, 642.0108032
2: -176.9501801, 240.2561340, -416.4230652, 536.1787720, -707.9809570, 647.4332275
3: -209.3154907, 271.8285217, -487.3614807, 608.0603027, -811.3602905, 750.5098267
4: -180.1105499, 275.6749573, -417.9467773, 614.2351685, -792.2343140, 685.4161987

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914763, upper bound: 398.8829047
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914763, upper bound: 398.8833715
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -182.5338745, 294.6768799, -379.4400024, 572.2309570, -749.7844849, 661.1018066
1: -200.3133240, 262.0391235, -414.8578186, 523.1924438, -717.9046631, 665.8549194
2: -200.4873199, 266.9307861, -413.7321472, 532.4129639, -727.3319702, 671.7182007
3: -235.6646729, 302.0430603, -484.1663208, 603.7816162, -833.0286255, 777.9891357
4: -202.8659515, 306.8235168, -415.1648865, 609.9190674, -810.2017212, 713.8225098

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917086, upper bound: 398.8807179
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8911956, upper bound: 398.8707591
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -379.2365417, 571.2230835, -164.8828735, 273.6366882, -639.9127808, 731.3446045
1: -414.5956116, 522.6860352, -181.5148926, 242.5971832, -645.4053345, 698.8177490
2: -413.4679565, 531.9111328, -181.4661713, 246.9006042, -650.8157959, 707.9234009
3: -483.7973022, 603.2556763, -214.6846161, 279.4453430, -754.4059448, 811.5939941
4: -414.8803406, 609.3031006, -184.8080750, 283.3780212, -689.8722534, 791.7625122

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829047, upper bound: 398.8914763
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829047, upper bound: 398.8914763
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -403.7687683, 606.2088623, -163.1987305, 270.5172119, -658.8758545, 762.6187744
1: -441.1639709, 555.2031860, -179.6420898, 239.8729553, -667.0197144, 727.4594116
2: -440.1089783, 565.1414185, -179.5790405, 244.1992493, -672.5858765, 737.3096313
3: -514.2994995, 640.9378662, -212.4424133, 276.3217468, -780.0216675, 844.6821289
4: -440.8140869, 647.1542358, -182.8298950, 280.2348328, -710.9708252, 826.3100586

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833715, upper bound: 398.8929370
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833715, upper bound: 398.8929370
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -379.4693604, 571.5990601, -382.1474609, 576.9548950, -934.4069214, 931.7293091
1: -414.8598328, 523.0347900, -417.8582153, 527.3723145, -920.6241455, 919.3082886
2: -413.7246399, 532.2597656, -416.7462769, 536.6198730, -930.4193115, 929.0993652
3: -484.1145325, 603.6630249, -487.7610474, 608.5759277, -1073.1617432, 1071.8328857
4: -415.1448059, 609.7048950, -418.2799072, 614.7428589, -1015.0192871, 1013.1253662

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8826892, upper bound: 398.8826892
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8826892, upper bound: 398.8828584
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -404.3253174, 607.0194092, -379.5080566, 572.3497314, -952.0193481, 962.3798828
1: -441.7807312, 555.9595947, -414.9341736, 523.2998657, -941.0614014, 947.2105103
2: -440.7174683, 565.8958130, -413.8071899, 532.5223999, -950.9684448, 957.7572632
3: -515.0227661, 641.8205566, -484.2586975, 603.9062500, -1097.4877930, 1104.0441895
4: -441.4168091, 648.0268555, -415.2423096, 610.0441284, -1034.7731934, 1047.0181885

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8706291, upper bound: 398.8793919
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
time: 0.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.33 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8927660, upper bound: 398.8927660
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8914763, upper bound: 398.8829047
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8914763, upper bound: 398.8833715
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8917086, upper bound: 398.8807179
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8911956, upper bound: 398.8707591
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8829047, upper bound: 398.8914763
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8829047, upper bound: 398.8914763
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8833715, upper bound: 398.8929370
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8833715, upper bound: 398.8929370
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8826892, upper bound: 398.8826892
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8826892, upper bound: 398.8828584
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8706291, upper bound: 398.8793919
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -160.8784637, 265.9215088, -426.7999268, 426.7999268
1: -177.0686951, 235.9393463, -177.0686951, 235.9393463, -413.0078430, 413.0078430
2: -176.9501801, 240.2561340, -176.9501801, 240.2561340, -417.2062988, 417.2062988
3: -209.3154907, 271.8285217, -209.3154907, 271.8285217, -481.1440125, 481.1440125
4: -180.1105499, 275.6749573, -180.1105499, 275.6749573, -455.7854919, 455.7854919

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8897257
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9175403, upper bound: 398.8907626
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -182.5338745, 294.6768799, -455.5553284, 448.4553528
1: -177.0686951, 235.9393463, -200.3133240, 262.0391235, -439.1076965, 436.2526245
2: -176.9501801, 240.2561340, -200.4873199, 266.9307861, -443.8809814, 440.7434692
3: -209.3154907, 271.8285217, -235.6646729, 302.0430603, -511.3585510, 507.4931946
4: -180.1105499, 275.6749573, -202.8659515, 306.8235168, -486.9340515, 478.5408325

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8897257
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9175403, upper bound: 398.8907626
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -182.5338745, 294.6768799, -160.8635864, 265.9017334, -448.4355469, 455.5404663
1: -200.3133240, 262.0391235, -177.0530396, 235.9219971, -436.2352905, 439.0921631
2: -200.4873199, 266.9307861, -176.9342804, 240.2380371, -440.7253418, 443.8650513
3: -235.6646729, 302.0430603, -209.2984161, 271.8085022, -507.4731750, 511.3414917
4: -202.8659515, 306.8235168, -180.0955811, 275.6540222, -478.5199585, 486.9190369

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8712051, upper bound: 398.8897633
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8906720, upper bound: 398.8906720
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -182.5338745, 294.6768799, -182.5338745, 294.6768799, -477.2107544, 477.2107544
1: -200.3133240, 262.0391235, -200.3133240, 262.0391235, -462.3524475, 462.3524475
2: -200.4873199, 266.9307861, -200.4873199, 266.9307861, -467.4180908, 467.4180908
3: -235.6646729, 302.0430603, -235.6646729, 302.0430603, -537.7077026, 537.7077026
4: -202.8659515, 306.8235168, -202.8659515, 306.8235168, -509.6894226, 509.6894226

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8712051, upper bound: 398.8897633
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8906720, upper bound: 398.8906720
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -379.1748962, 571.1436157, -727.3558350, 632.0898438
1: -177.0686951, 235.9393463, -414.5252075, 522.6077271, -694.3588257, 638.7440186
2: -176.9501801, 240.2561340, -413.4006958, 531.8346558, -703.3927612, 644.1520386
3: -209.3154907, 271.8285217, -483.7154846, 603.1632690, -806.1777954, 746.6455078
4: -180.1105499, 275.6749573, -414.8119202, 609.2136230, -787.0112305, 682.0732422

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8896945, upper bound: 398.8820126
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9138094, upper bound: 398.8822417
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -403.6163330, 606.0364990, -760.1996460, 654.0710449
1: -177.0686951, 235.9393463, -440.9938049, 555.0298462, -724.7606201, 662.9593506
2: -176.9501801, 240.2561340, -439.9440308, 564.9731445, -734.5593872, 668.5034790
3: -209.3154907, 271.8285217, -514.1059570, 640.7334595, -841.3820190, 775.2996216
4: -180.1105499, 275.6749573, -440.6520691, 646.9560547, -823.4241943, 706.2170410

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8896945, upper bound: 398.8820161
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9138094, upper bound: 398.8822494
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -178.1414642, 288.0104370, -366.1704407, 550.7203369, -723.3447266, 640.3409424
1: -195.4478149, 255.8342590, -400.0985107, 503.4113770, -692.6652222, 644.0228882
2: -195.6358948, 260.6507263, -399.0382080, 512.4703369, -702.0065918, 649.8712769
3: -229.9639282, 294.8898010, -466.6648865, 581.1901245, -804.1193237, 752.6239014
4: -197.9047699, 299.5965576, -400.0324707, 587.0120850, -781.8269653, 690.7924805

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650830, upper bound: 398.8778737
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891087, upper bound: 398.8781796
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -180.8689575, 291.6378784, -375.6971436, 566.0205688, -741.8408813, 654.2834473
1: -198.4547729, 259.2663574, -410.7552185, 517.6458740, -710.4392090, 658.9637451
2: -198.6274109, 264.1130981, -409.5889893, 526.8142700, -719.8651733, 664.7548218
3: -233.4275970, 298.8744507, -479.3229980, 597.4433594, -824.3577271, 769.9799805
4: -200.9393921, 303.6443787, -411.0038757, 603.5349731, -801.8879395, 706.5418701

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8633737, upper bound: 398.8673086
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8885138, upper bound: 398.8674981
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -379.1748962, 571.1436157, -160.8784637, 265.9215088, -632.0898438, 727.3558960
1: -414.5252075, 522.6077271, -177.0686951, 235.9393463, -638.7440186, 694.3588257
2: -413.4006958, 531.8347168, -176.9501801, 240.2561340, -644.1520996, 703.3927612
3: -483.7154846, 603.1632690, -209.3154907, 271.8285217, -746.6455078, 806.1777954
4: -414.8119202, 609.2136230, -180.1105499, 275.6749573, -682.0732422, 787.0112305

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793296, upper bound: 398.8894499
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653766, upper bound: 398.8870136
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8819324, upper bound: 398.8893376
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -379.4059448, 571.4886475, -182.5338745, 294.6768799, -660.8408813, 748.8903198
1: -414.7885437, 522.9348145, -200.3133240, 262.0391235, -665.5833130, 717.5222168
2: -413.6546326, 532.1580200, -200.4873199, 266.9307861, -671.4376831, 726.9544678
3: -484.0284119, 603.5470581, -235.6646729, 302.0430603, -777.6943970, 832.6524658
4: -415.0726318, 609.5883179, -202.8659515, 306.8235168, -713.5762939, 809.7692261

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793296, upper bound: 398.8894499
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653766, upper bound: 398.8870136
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8819324, upper bound: 398.8893376
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -403.6910706, 606.1075439, -160.8635864, 265.9017334, -654.1166992, 760.2561035
1: -441.0788269, 555.1073608, -177.0530396, 235.9219971, -663.0163574, 724.8225098
2: -440.0246277, 565.0464478, -176.9342804, 240.2380371, -668.5575562, 734.6170044
3: -514.2019043, 640.8256836, -209.2984161, 271.8085022, -775.3649902, 841.4569702
4: -440.7325745, 647.0432129, -180.0955811, 275.6540222, -706.2707520, 823.4962158

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8807179, upper bound: 398.8917085
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707586, upper bound: 398.8911956
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -404.1617737, 606.7189331, -182.5338745, 294.6768799, -683.1016846, 782.0670776
1: -441.5959473, 555.6885376, -200.3133240, 262.0391235, -690.1036987, 748.2526855
2: -440.5356750, 565.6223145, -200.4873199, 266.9307861, -696.0951538, 758.4447021
3: -514.7957153, 641.5060425, -235.6646729, 302.0430603, -806.6942749, 868.2420044
4: -441.2285767, 647.7154541, -202.8659515, 306.8235168, -738.0153198, 846.5652466

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8807179, upper bound: 398.8917086
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707586, upper bound: 398.8911956
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -379.4693604, 571.5990601, -379.4693604, 571.5990601, -928.7586060, 928.7586060
1: -414.8598328, 523.0347900, -414.8598328, 523.0347900, -916.0402222, 916.0402222
2: -413.7246399, 532.2597656, -413.7246399, 532.2597656, -925.8160400, 925.8160400
3: -484.1145325, 603.6630249, -484.1145325, 603.6630249, -1067.9649658, 1067.9648438
4: -415.1448059, 609.7048950, -415.1448059, 609.7048950, -1009.7804565, 1009.7804565

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649894, upper bound: 398.8811142
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817202, upper bound: 398.8816149
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -379.4693604, 571.5990601, -404.3253174, 607.0194092, -962.1144409, 951.1172485
1: -414.8598328, 523.0347900, -441.7807312, 555.9595947, -946.9337158, 940.6716919
2: -413.7246399, 532.2597656, -440.7174683, 565.8958130, -957.4715576, 950.5836792
3: -484.1145325, 603.6630249, -515.0227661, 641.8205566, -1103.7432861, 1097.1033936
4: -415.1448059, 609.7048950, -441.4168091, 648.0268555, -1046.7667236, 1034.3321533

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649894, upper bound: 398.8811142
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817202, upper bound: 398.8816149
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -399.3267822, 599.4274292, -366.4153137, 551.0619507, -925.0769653, 940.9066162
1: -436.2094116, 548.7506104, -400.3619690, 503.7232361, -915.3088379, 924.6779785
2: -435.2132263, 558.6291504, -399.3040771, 512.7872925, -925.0793457, 935.2451782
3: -508.4669800, 633.5365601, -466.9674683, 581.5515137, -1067.9227295, 1077.8663330
4: -435.7906189, 639.6604004, -400.2857666, 587.3818359, -1005.9447021, 1023.1235352

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -402.9755249, 604.5760498, -376.2707520, 566.7064209, -944.9304199, 956.5863647
1: -440.2774658, 553.7827148, -411.3579712, 518.2764282, -934.4595947, 941.3612061
2: -439.2059937, 563.7019043, -410.2055054, 527.4533081, -944.3400879, 951.8778687
3: -513.2021484, 639.3332520, -479.9881897, 598.1775513, -1089.8378906, 1097.2073975
4: -439.8457031, 645.5326538, -411.5643311, 604.2880859, -1027.4553223, 1040.8131104

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.80 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8897257
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.9175403, upper bound: 398.8907626
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8897257
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.9175403, upper bound: 398.8907626
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8712051, upper bound: 398.8897633
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8906720, upper bound: 398.8906720
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8712051, upper bound: 398.8897633
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8906720, upper bound: 398.8906720
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8896945, upper bound: 398.8820126
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.9138094, upper bound: 398.8822417
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8896945, upper bound: 398.8820161
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.9138094, upper bound: 398.8822494
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8650830, upper bound: 398.8778737
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8891087, upper bound: 398.8781796
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8633737, upper bound: 398.8673086
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8885138, upper bound: 398.8674981
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8653766, upper bound: 398.8870136
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8819324, upper bound: 398.8893376
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8653766, upper bound: 398.8870136
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8819324, upper bound: 398.8893376
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8807179, upper bound: 398.8917085
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8707586, upper bound: 398.8911956
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8807179, upper bound: 398.8917086
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8707586, upper bound: 398.8911956
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8649894, upper bound: 398.8811142
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8817202, upper bound: 398.8816149
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8649894, upper bound: 398.8811142
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8817202, upper bound: 398.8816149
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 0, lower bound: -398.8695199, upper bound: 398.8695199

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -160.6211395, 265.4533081, -415.2980652, 405.3330383
1: -164.6911926, 217.2070312, -176.7797394, 235.5208893, -400.2120972, 393.9867554
2: -164.6050262, 220.8984070, -176.6647949, 239.8218536, -404.4268494, 397.5631714
3: -193.9470978, 250.2816925, -208.9590149, 271.3406067, -465.2877197, 459.2406006
4: -167.2314453, 253.8219147, -179.8074036, 275.1850891, -442.4165344, 433.6293335

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8983241, upper bound: 398.8983241
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8983241, upper bound: 398.9243332
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -160.8784637, 265.9215088, -425.2685547, 424.0083313
1: -175.3673859, 233.4939575, -177.0686951, 235.9393463, -411.3066711, 410.5625000
2: -175.2652588, 237.8358307, -176.9501801, 240.2561340, -415.5213928, 414.7859497
3: -207.2741852, 269.0414734, -209.3154907, 271.8285217, -479.1026611, 478.3569641
4: -178.3538513, 272.8920898, -180.1105499, 275.6749573, -454.0287476, 453.0026245

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9243332, upper bound: 398.8984899
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9243332, upper bound: 398.9247252
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -182.2581024, 294.1833801, -444.0281372, 426.9700012
1: -164.6911926, 217.2070312, -200.0035400, 261.6096802, -426.3008728, 417.2105713
2: -164.6050262, 220.8984070, -200.1821289, 266.4823303, -431.0873108, 421.0805359
3: -193.9470978, 250.2816925, -235.2860718, 301.5429382, -495.4900513, 485.5676575
4: -167.2314453, 253.8219147, -202.5496979, 306.3214417, -473.5528870, 456.3716125

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8712270
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8897257
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -182.5338745, 294.6768799, -454.0239563, 445.6637573
1: -175.3673859, 233.4939575, -200.3133240, 262.0391235, -437.4064941, 433.8072205
2: -175.2652588, 237.8358307, -200.4873199, 266.9307861, -442.1959534, 438.3230896
3: -207.2741852, 269.0414734, -235.6646729, 302.0430603, -509.3171997, 504.7061462
4: -178.3538513, 272.8920898, -202.8659515, 306.8235168, -485.1773071, 475.7580261

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9163761, upper bound: 398.8713342
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9163761, upper bound: 398.8907626
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -171.1241913, 273.7840576, -160.6062927, 265.4335938, -436.5577698, 434.3903503
1: -187.5600433, 243.4932709, -176.7641602, 235.5035553, -423.0635986, 420.2574158
2: -187.7182159, 247.8797760, -176.6488953, 239.8036804, -427.5219116, 424.5285645
3: -220.0472717, 280.7072754, -208.9419556, 271.3206482, -491.3678284, 489.6492310
4: -189.6266632, 285.1177368, -179.7924042, 275.1642151, -464.7908325, 464.9100647

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8712270, upper bound: 398.8931770
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8712270, upper bound: 398.9163761
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -181.0280762, 291.8794556, -160.8635864, 265.9017334, -446.9297791, 452.7430420
1: -198.6312103, 259.5796204, -177.0530396, 235.9219971, -434.5531311, 436.6326599
2: -198.8130798, 264.4493713, -176.9342804, 240.2380371, -439.0510864, 441.3836670
3: -233.6219635, 299.2090454, -209.2984161, 271.8085022, -505.4304810, 508.5074463
4: -201.1061249, 303.9932556, -180.0955811, 275.6540222, -476.7601318, 484.0888062

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897257, upper bound: 398.8931770
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897257, upper bound: 398.9175403
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -171.1241913, 273.7840576, -182.2581024, 294.1833801, -465.3075562, 456.0421753
1: -187.5600433, 243.4932709, -200.0035400, 261.6096802, -449.1697388, 443.4967957
2: -187.7182159, 247.8797760, -200.1821289, 266.4823303, -454.2005005, 448.0618896
3: -220.0472717, 280.7072754, -235.2860718, 301.5429382, -521.5902100, 515.9933472
4: -189.6266632, 285.1177368, -202.5496979, 306.3214417, -495.9480896, 487.6673889

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711615, upper bound: 398.8711615
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711615, upper bound: 398.8897633
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -181.0280762, 291.8794556, -182.5338745, 294.6768799, -475.7049561, 474.4133301
1: -198.6312103, 259.5796204, -200.3133240, 262.0391235, -460.6702271, 459.8929443
2: -198.8130798, 264.4493713, -200.4873199, 266.9307861, -465.7437744, 464.9367065
3: -233.6219635, 299.2090454, -235.6646729, 302.0430603, -535.6648560, 534.8737183
4: -201.1061249, 303.9932556, -202.8659515, 306.8235168, -507.9296265, 506.8591919

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897257, upper bound: 398.8712051
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897257, upper bound: 398.8906720
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -378.4824829, 570.0882568, -715.1665649, 609.4088745
1: -164.6911926, 217.2070312, -413.7450867, 521.6286011, -680.7519531, 618.7521973
2: -164.6050262, 220.8984070, -412.6389465, 530.8585815, -690.0514526, 623.4518433
3: -193.9470978, 250.2816925, -482.7886047, 602.0246582, -789.3858643, 723.5718384
4: -167.2314453, 253.8219147, -414.0307312, 608.0818481, -772.5735474, 658.9248047

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836999, upper bound: 398.8787903
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895129, upper bound: 398.8653316
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895129, upper bound: 398.8820126
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -379.1231995, 571.0697021, -725.7379150, 629.3230591
1: -175.3673859, 233.4939575, -414.4682922, 522.5390015, -692.6026611, 636.3098755
2: -175.2652588, 237.8358307, -413.3443298, 531.7660522, -701.6345215, 641.7280884
3: -207.2741852, 269.0414734, -483.6491394, 603.0826416, -804.0720825, 743.8072510
4: -178.3538513, 272.8920898, -414.7565613, 609.1337280, -785.1750488, 679.2754517

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9054922, upper bound: 398.8793036
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9125891, upper bound: 398.8654756
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9125891, upper bound: 398.8822417
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -401.6120911, 603.0472412, -746.1651611, 630.1575317
1: -164.6911926, 217.2070312, -438.7982483, 552.2455444, -709.4321289, 641.6238403
2: -164.6050262, 220.8984070, -437.7508850, 562.1589966, -719.4671021, 646.4432373
3: -193.9470978, 250.2816925, -511.5436707, 637.5021362, -822.5906372, 750.6611328
4: -167.2314453, 253.8219147, -438.4714355, 643.7568970, -806.9758301, 681.7351074

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8843892, upper bound: 398.8789833
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787500, upper bound: 398.8673037
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -403.5564575, 605.9521484, -758.5717163, 651.2966919
1: -175.3673859, 233.4939575, -440.9283447, 554.9515991, -722.9951782, 660.5172119
2: -175.2652588, 237.8358307, -439.8789062, 564.8949585, -732.7920532, 666.0716553
3: -207.2741852, 269.0414734, -514.0300293, 640.6420898, -839.2659912, 772.4523315
4: -178.3538513, 272.8920898, -440.5889282, 646.8651733, -821.5771484, 703.4119873

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9062117, upper bound: 398.8794812
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9052287, upper bound: 398.8674883
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -166.5733948, 266.7813416, -365.1292725, 548.9536133, -710.0250854, 617.0494385
1: -182.5161591, 237.0055695, -398.9316101, 501.8201294, -678.0568237, 623.3322144
2: -182.6948242, 241.2886963, -397.8700562, 510.8621826, -687.4041138, 628.5454712
3: -214.1276398, 273.2288513, -465.2456360, 579.3444824, -786.2781982, 728.7240601
4: -184.4762421, 277.5671692, -398.8352356, 585.1455078, -766.2267456, 667.1875610

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650209, upper bound: 398.8584290
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8778737
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -176.6123657, 285.1778259, -366.1166992, 550.6369019, -721.7287598, 637.5336914
1: -193.7355957, 253.3409424, -400.0395813, 503.3353882, -690.8847046, 641.5363159
2: -193.9372253, 258.1333923, -398.9792786, 512.3933716, -700.2192383, 647.3748169
3: -227.8819580, 292.0210571, -466.5950928, 581.1021729, -801.9642334, 749.7627563
4: -196.1167755, 296.7211914, -399.9744568, 586.9224854, -779.9803467, 687.9548340

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8890656, upper bound: 398.8586497
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8890656, upper bound: 398.8781796
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -169.6011047, 270.9648438, -375.1256104, 565.0602417, -729.7223511, 632.0754395
1: -185.8597412, 240.9204102, -410.1157837, 516.7808838, -696.9205933, 639.3472900
2: -186.0190887, 245.3761444, -408.9558411, 525.9419556, -706.3983765, 644.5231934
3: -217.9990540, 277.7748108, -478.5509949, 596.4431152, -807.8000488, 747.3687134
4: -187.8750916, 282.1921692, -410.3530579, 602.5275879, -787.5582886, 684.0355225

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600712, upper bound: 398.8646031
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8630903, upper bound: 398.8662326
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -179.5081482, 289.1158142, -375.6168823, 565.9110718, -740.3851929, 651.7807007
1: -196.9323730, 257.0628357, -410.6683655, 517.5450439, -708.8351440, 656.7636719
2: -197.1161957, 261.8901062, -409.5022583, 526.7123413, -718.2429810, 662.5515747
3: -231.5811310, 296.3304138, -479.2236023, 597.3260498, -822.4193115, 767.4406128
4: -199.3473969, 301.1004028, -410.9201660, 603.4162598, -800.2237549, 704.0119019

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8868516, upper bound: 398.8646960
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870890, upper bound: 398.8664066
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -368.6454773, 551.7455444, -160.6211395, 265.4533081, -620.5826416, 706.5190430
1: -402.6379395, 505.3999329, -176.7797394, 235.5208893, -626.0839233, 675.8805542
2: -401.6080322, 514.3607788, -176.6647949, 239.8218536, -631.4182129, 684.5424805
3: -469.1020813, 583.5023804, -208.9590149, 271.3406067, -731.1937866, 784.9718018
4: -402.2611084, 589.0947266, -179.8074036, 275.1850891, -668.5664673, 765.7952881

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653317, upper bound: 398.8895129
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653317, upper bound: 398.9125891
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -377.8083191, 568.4946289, -160.8784637, 265.9215088, -630.5977783, 724.7041626
1: -412.9947205, 520.3738403, -177.0686951, 235.9393463, -637.1276855, 692.0916138
2: -411.8705139, 529.5405273, -176.9501801, 240.2561340, -642.4829102, 701.0846558
3: -481.8311157, 600.6121826, -209.3154907, 271.8285217, -744.6835938, 803.5907593
4: -413.2162170, 606.6295166, -180.1105499, 275.6749573, -680.4154053, 784.3947144

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820126, upper bound: 398.8896945
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820126, upper bound: 398.9138094
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -368.9599609, 552.1882935, -182.2581024, 294.1833801, -649.3947754, 728.1498413
1: -402.9881897, 505.8174133, -200.0035400, 261.6096802, -652.9993896, 699.1242676
2: -401.9517212, 514.7746582, -200.1821289, 266.4823303, -658.7807617, 708.1738281
3: -469.5090637, 583.9916382, -235.2860718, 301.5429382, -762.3327026, 811.5373535
4: -402.6020813, 589.5740356, -202.5496979, 306.3214417, -700.1384888, 788.6510010

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652890, upper bound: 398.8696138
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652890, upper bound: 398.8870136
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -378.0603943, 568.8607788, -182.5338745, 294.6768799, -659.3674927, 746.2596436
1: -413.2810364, 520.7225342, -200.3133240, 262.0391235, -663.9873047, 715.2766113
2: -412.1470032, 529.8846436, -200.4873199, 266.9307861, -669.7889404, 724.6676025
3: -482.1694641, 601.0217285, -235.6646729, 302.0430603, -775.7553711, 830.0911865
4: -413.4983215, 607.0289917, -202.8659515, 306.8235168, -711.9384155, 807.1778564

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817932, upper bound: 398.8703722
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817932, upper bound: 398.8893376
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -391.8916626, 586.9262085, -157.1102142, 260.1480408, -635.6767578, 736.8384399
1: -427.9274292, 537.6356201, -172.8681183, 230.5657806, -643.6602173, 702.6765137
2: -426.9412537, 547.4508057, -172.7858582, 234.7873840, -649.1328125, 712.3847656
3: -498.6222534, 620.8376465, -204.3525238, 265.6392517, -753.0127563, 816.0086670
4: -427.2668457, 626.7052612, -175.8606110, 269.3927917, -685.9307251, 798.4517822

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8789833, upper bound: 398.8843892
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794812, upper bound: 398.9062117
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -400.0044556, 600.0868530, -158.9571991, 262.5394287, -647.0247803, 752.2573853
1: -437.0305176, 549.7687988, -174.9351959, 232.8255768, -655.8234863, 717.2944946
2: -435.9400330, 559.6620483, -174.8102570, 237.1615601, -661.3496704, 727.0730591
3: -509.4260864, 634.7102051, -206.7514648, 268.2824707, -766.9553833, 832.6855469
4: -436.6458130, 640.8367310, -177.8864288, 272.1412659, -698.6567383, 815.0442505

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673037, upper bound: 398.8787500
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674883, upper bound: 398.9052287
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -392.3059082, 587.5030518, -178.1414642, 288.0104370, -663.6881104, 757.9486694
1: -428.3886414, 538.1853027, -195.4478149, 255.8342590, -669.8133545, 725.3562622
2: -427.3944397, 547.9956665, -195.6358948, 260.6507263, -675.7446289, 735.4911499
3: -499.1635437, 621.4790039, -229.9639282, 294.8898010, -783.2011719, 841.9804077
4: -427.7189941, 627.3391113, -197.9047699, 299.5965576, -716.6210327, 820.7528076

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8778737, upper bound: 398.8652645
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8781796, upper bound: 398.8891087
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -400.5648804, 600.7958984, -180.8689575, 291.6378784, -676.4132080, 774.3828125
1: -437.6405029, 550.4426270, -198.4547729, 259.2663574, -683.3488770, 741.0778809
2: -436.5469971, 560.3316650, -198.6274109, 264.1130981, -689.2745972, 751.2681274
3: -510.1205139, 635.4976196, -233.4275970, 298.8744507, -798.8363647, 859.8926392
4: -437.2267761, 641.6190796, -200.9393921, 303.6443787, -730.8871460, 838.5192871

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673086, upper bound: 398.8633736
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674981, upper bound: 398.8885138
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -369.0663452, 552.3657227, -379.2254639, 571.1682739, -917.4516602, 908.1036377
1: -403.1070862, 505.9794922, -414.5870667, 522.6541748, -903.5885010, 897.7545776
2: -402.0683594, 514.9389648, -413.4532471, 531.8738403, -913.2996216, 907.1213989
3: -469.6511230, 584.1793823, -483.7819824, 603.2278442, -1052.7602539, 1046.9766846
4: -402.7211609, 589.7617798, -414.8599243, 609.2610474, -996.4653320, 988.7813721

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646138, upper bound: 398.8646138
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646138, upper bound: 398.8813208
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -378.1610107, 569.0414429, -379.4693604, 571.5990601, -927.3214111, 926.1946411
1: -413.3947754, 520.8857422, -414.8598328, 523.0347900, -914.4857178, 913.8551636
2: -412.2584534, 530.0501709, -413.7246399, 532.2597656, -924.2080688, 923.5906372
3: -482.3082886, 601.2113037, -484.1145325, 603.6630249, -1066.0776367, 1065.4739990
4: -413.6139221, 607.2186279, -415.1448059, 609.7048950, -1008.1848755, 1007.2605591

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813208, upper bound: 398.8649894
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813208, upper bound: 398.8817202
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -369.0663452, 552.3657227, -404.0748901, 606.5832520, -950.8054810, 930.4558105
1: -403.1070862, 505.9794922, -441.5001831, 555.5733032, -934.4794312, 922.3787842
2: -402.0683594, 514.9389648, -440.4397583, 565.5047607, -944.9524536, 931.8822632
3: -469.6511230, 584.1793823, -514.6816406, 641.3787842, -1088.5368652, 1076.1052246
4: -402.7211609, 589.7617798, -441.1240234, 647.5750122, -1033.4472656, 1013.3257446

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652216, upper bound: 398.8647816
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652216, upper bound: 398.8811142
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -378.1610107, 569.0414429, -404.3253174, 607.0194092, -960.6771240, 948.5532837
1: -413.3947754, 520.8857422, -441.7807312, 555.9595947, -945.3793335, 938.4866333
2: -412.2584534, 530.0501709, -440.7174683, 565.8958130, -955.8635864, 948.3582764
3: -482.3082886, 601.2113037, -515.0227661, 641.8205566, -1101.8562012, 1094.6125488
4: -413.6139221, 607.2186279, -441.4168091, 648.0268555, -1045.1713867, 1031.8121338

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817089, upper bound: 398.8651760
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817089, upper bound: 398.8816149
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -392.9279175, 588.3497314, -366.4153137, 551.0619507, -918.1304932, 929.5374146
1: -429.0569458, 538.9698486, -400.3619690, 503.7232361, -907.6820068, 914.6727295
2: -428.0704956, 548.7850342, -399.3040771, 512.7872925, -917.4761963, 925.1756592
3: -499.9305420, 622.3893433, -466.9674683, 581.5515137, -1059.0728760, 1066.4581299
4: -428.3645935, 628.2581787, -400.2857666, 587.3818359, -998.2807007, 1011.5528564

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673240, upper bound: 398.8739456
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8692898, upper bound: 398.8788475
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -401.4359131, 601.8801880, -366.4153137, 551.0619507, -926.8894653, 943.1508789
1: -438.5582275, 551.4457397, -400.3619690, 503.7232361, -917.3627319, 927.1623535
2: -437.4852295, 561.3435669, -399.3040771, 512.7872925, -927.0907593, 937.7385254
3: -511.1441040, 636.6628418, -466.9674683, 581.5515137, -1070.3891602, 1080.7650146
4: -438.0916748, 642.8013306, -400.2857666, 587.3818359, -1008.0709839, 1026.1037598

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673240, upper bound: 398.8739456
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8692898, upper bound: 398.8788475
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -392.6799622, 587.8555298, -376.2707520, 566.7064209, -933.6487427, 939.0884399
1: -428.7810059, 538.5103760, -411.3579712, 518.2764282, -922.0313721, 925.3693848
2: -427.7910461, 548.3222046, -410.2055054, 527.4533081, -931.9296875, 935.7531738
3: -499.5847168, 621.8666382, -479.9881897, 598.1775513, -1075.4565430, 1079.0456543
4: -428.0610657, 627.7503052, -411.5643311, 604.2880859, -1014.9284058, 1022.4093018

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673169, upper bound: 398.8656553
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8680063, upper bound: 398.8680063
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -401.4359131, 601.8801880, -376.2707520, 566.7064209, -943.3403931, 953.8669434
1: -438.5582275, 551.4457397, -411.3579712, 518.2764282, -932.7048340, 939.0283813
2: -437.4852295, 561.3435669, -410.2055054, 527.4533081, -942.5773926, 949.5294800
3: -511.1441040, 636.6628418, -479.9881897, 598.1775513, -1087.7674561, 1094.5274658
4: -438.0916748, 642.8013306, -411.5643311, 604.2880859, -1025.6809082, 1038.1029053

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673169, upper bound: 398.8656553
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8680063, upper bound: 398.8680063
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.60 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8983241, upper bound: 398.8983241
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8983241, upper bound: 398.9243332
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9243332, upper bound: 398.8984899
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9243332, upper bound: 398.9247252
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8712270
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8931770, upper bound: 398.8897257
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9163761, upper bound: 398.8713342
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9163761, upper bound: 398.8907626
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8712270, upper bound: 398.8931770
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8712270, upper bound: 398.9163761
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8897257, upper bound: 398.8931770
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8897257, upper bound: 398.9175403
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8711615, upper bound: 398.8711615
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8711615, upper bound: 398.8897633
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8897257, upper bound: 398.8712051
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8897257, upper bound: 398.8906720
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8895129, upper bound: 398.8653316
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8895129, upper bound: 398.8820126
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9125891, upper bound: 398.8654756
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9125891, upper bound: 398.8822417
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8843892, upper bound: 398.8789833
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8787500, upper bound: 398.8673037
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9062117, upper bound: 398.8794812
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.9052287, upper bound: 398.8674883
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8650209, upper bound: 398.8584290
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8778737
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8890656, upper bound: 398.8586497
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8890656, upper bound: 398.8781796
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8600712, upper bound: 398.8646031
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8630903, upper bound: 398.8662326
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8868516, upper bound: 398.8646960
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8870890, upper bound: 398.8664066
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8653317, upper bound: 398.8895129
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8653317, upper bound: 398.9125891
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8820126, upper bound: 398.8896945
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8820126, upper bound: 398.9138094
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8652890, upper bound: 398.8696138
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8652890, upper bound: 398.8870136
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8817932, upper bound: 398.8703722
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8817932, upper bound: 398.8893376
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8789833, upper bound: 398.8843892
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8794812, upper bound: 398.9062117
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8673037, upper bound: 398.8787500
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8674883, upper bound: 398.9052287
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8778737, upper bound: 398.8652645
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8781796, upper bound: 398.8891087
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8673086, upper bound: 398.8633736
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8674981, upper bound: 398.8885138
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8646138, upper bound: 398.8646138
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8646138, upper bound: 398.8813208
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8813208, upper bound: 398.8649894
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8813208, upper bound: 398.8817202
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8652216, upper bound: 398.8647816
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8652216, upper bound: 398.8811142
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8817089, upper bound: 398.8651760
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8817089, upper bound: 398.8816149
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8673240, upper bound: 398.8739456
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8692898, upper bound: 398.8788475
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8673240, upper bound: 398.8739456
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8692898, upper bound: 398.8788475
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8673169, upper bound: 398.8656553
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8680063, upper bound: 398.8680063
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8673169, upper bound: 398.8656553
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -398.8680063, upper bound: 398.8680063

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -149.8447571, 244.7119904, -394.5567017, 394.5566711
1: -164.6911926, 217.2070312, -164.6911926, 217.2070312, -381.8982239, 381.8982239
2: -164.6050262, 220.8984070, -164.6050262, 220.8984070, -385.5033875, 385.5033875
3: -193.9470978, 250.2816925, -193.9470978, 250.2816925, -444.2286987, 444.2286987
4: -167.2314453, 253.8219147, -167.2314453, 253.8219147, -421.0533447, 421.0533447

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8780922, upper bound: 398.8910622
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8971743, upper bound: 398.8971743
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -159.3470764, 263.1299133, -412.9746704, 404.0589600
1: -164.6911926, 217.2070312, -175.3673859, 233.4939575, -398.1851501, 392.5744019
2: -164.6050262, 220.8984070, -175.2652588, 237.8358307, -402.4407349, 396.1636047
3: -193.9470978, 250.2816925, -207.2741852, 269.0414734, -462.9885864, 457.5557556
4: -167.2314453, 253.8219147, -178.3538513, 272.8920898, -440.1235352, 432.1757812

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8780922, upper bound: 398.9159720
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8971743, upper bound: 398.9239766
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -149.8447571, 244.7119904, -404.0589600, 412.9746704
1: -175.3673859, 233.4939575, -164.6911926, 217.2070312, -392.5744019, 398.1851196
2: -175.2652588, 237.8358307, -164.6050262, 220.8984070, -396.1636047, 402.4407349
3: -207.2741852, 269.0414734, -193.9470978, 250.2816925, -457.5557556, 462.9885864
4: -178.3538513, 272.8920898, -167.2314453, 253.8219147, -432.1757812, 440.1235352

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8994488, upper bound: 398.8911733
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9239765, upper bound: 398.8974361
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -159.3470764, 263.1299133, -422.4769592, 422.4769592
1: -175.3673859, 233.4939575, -175.3673859, 233.4939575, -408.8612976, 408.8612976
2: -175.2652588, 237.8358307, -175.2652588, 237.8358307, -413.1009827, 413.1009827
3: -207.2741852, 269.0414734, -207.2741852, 269.0414734, -476.3156433, 476.3156433
4: -178.3538513, 272.8920898, -178.3538513, 272.8920898, -451.2459106, 451.2459106

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8994489, upper bound: 398.9158200
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9239765, upper bound: 398.9239204
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -171.1241913, 273.7840576, -423.6288147, 415.8360901
1: -164.6911926, 217.2070312, -187.5600433, 243.4932709, -408.1844177, 404.7670898
2: -164.6050262, 220.8984070, -187.7182159, 247.8797760, -412.4847717, 408.6166382
3: -193.9470978, 250.2816925, -220.0472717, 280.7072754, -474.6543579, 470.3288574
4: -167.2314453, 253.8219147, -189.6266632, 285.1177368, -452.3491821, 443.4485779

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8725274, upper bound: 398.8650562
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919743, upper bound: 398.8698238
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -181.0280762, 291.8794556, -441.7242126, 425.7399597
1: -164.6911926, 217.2070312, -198.6312103, 259.5796204, -424.2708130, 415.8381958
2: -164.6050262, 220.8984070, -198.8130798, 264.4493713, -429.0543823, 419.7114563
3: -193.9470978, 250.2816925, -233.6219635, 299.2090454, -493.1561279, 483.9036560
4: -167.2314453, 253.8219147, -201.1061249, 303.9932556, -471.2247009, 454.9280396

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8725274, upper bound: 398.8807459
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919743, upper bound: 398.8891460
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -171.1241913, 273.7840576, -433.1311340, 434.2540894
1: -175.3673859, 233.4939575, -187.5600433, 243.4932709, -418.8606262, 421.0540161
2: -175.2652588, 237.8358307, -187.7182159, 247.8797760, -423.1449890, 425.5540161
3: -207.2741852, 269.0414734, -220.0472717, 280.7072754, -487.9814453, 489.0887451
4: -178.3538513, 272.8920898, -189.6266632, 285.1177368, -463.4715271, 462.5187378

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8904131, upper bound: 398.8653017
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9154555, upper bound: 398.8699556
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -181.0280762, 291.8794556, -451.2265320, 444.1579590
1: -175.3673859, 233.4939575, -198.6312103, 259.5796204, -434.9470215, 432.1250305
2: -175.2652588, 237.8358307, -198.8130798, 264.4493713, -439.7146301, 436.6488037
3: -207.2741852, 269.0414734, -233.6219635, 299.2090454, -506.4831848, 502.6634521
4: -178.3538513, 272.8920898, -201.1061249, 303.9932556, -482.3470764, 473.9982300

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8904131, upper bound: 398.8810251
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9154555, upper bound: 398.8903988
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -171.1241913, 273.7840576, -149.8200073, 244.6792603, -415.8034668, 423.6040649
1: -187.5600433, 243.4932709, -164.6651764, 217.1784058, -404.7384644, 408.1584473
2: -187.7182159, 247.8797760, -164.5785828, 220.8682709, -408.5864258, 412.4583435
3: -220.0472717, 280.7072754, -193.9185028, 250.2486725, -470.2958984, 474.6257935
4: -189.6266632, 285.1177368, -167.2063751, 253.7871246, -443.4137878, 452.3240967

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8516072, upper bound: 398.8886582
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8516072, upper bound: 398.8933591
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -171.1241913, 273.7840576, -159.3321838, 263.1101379, -434.2343140, 433.1162415
1: -187.5600433, 243.4932709, -175.3517456, 233.4766235, -421.0366821, 418.8450012
2: -187.7182159, 247.8797760, -175.2492981, 237.8176880, -425.5358887, 423.1290894
3: -220.0472717, 280.7072754, -207.2570496, 269.0214233, -489.0686646, 487.9643250
4: -189.6266632, 285.1177368, -178.3388214, 272.8711243, -462.4977722, 463.4565430

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8516072, upper bound: 398.9142135
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711674, upper bound: 398.9163761
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -181.0280762, 291.8794556, -149.8200073, 244.6792603, -425.7073059, 441.6994629
1: -198.6312103, 259.5796204, -164.6651764, 217.1784058, -415.8095703, 424.2448120
2: -198.8130798, 264.4493713, -164.5785828, 220.8682709, -419.6812744, 429.0279541
3: -233.6219635, 299.2090454, -193.9185028, 250.2486725, -483.8706360, 493.1275635
4: -201.1061249, 303.9932556, -167.2063751, 253.7871246, -454.8932495, 471.1996460

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8893602, upper bound: 398.8893467
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8882486, upper bound: 398.8893518
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -181.0280762, 291.8794556, -159.3321838, 263.1101379, -444.1381836, 451.2116394
1: -198.6312103, 259.5796204, -175.3517456, 233.4766235, -432.1077881, 434.9313660
2: -198.8130798, 264.4493713, -175.2492981, 237.8176880, -436.6307373, 439.6986694
3: -233.6219635, 299.2090454, -207.2570496, 269.0214233, -502.6433716, 506.4660950
4: -201.1061249, 303.9932556, -178.3388214, 272.8711243, -473.9772339, 482.3320923

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8893603, upper bound: 398.9142401
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8882486, upper bound: 398.9140139
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -171.1241913, 273.7840576, -171.1241913, 273.7840576, -444.9082642, 444.9082642
1: -187.5600433, 243.4932709, -187.5600433, 243.4932709, -431.0532837, 431.0532837
2: -187.7182159, 247.8797760, -187.7182159, 247.8797760, -435.5979919, 435.5979919
3: -220.0472717, 280.7072754, -220.0472717, 280.7072754, -500.7545471, 500.7545471
4: -189.6266632, 285.1177368, -189.6266632, 285.1177368, -474.7443542, 474.7443542

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8512354, upper bound: 398.8656831
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8710937, upper bound: 398.8710971
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -171.1241913, 273.7840576, -181.0280762, 291.8794556, -463.0036621, 454.8121338
1: -187.5600433, 243.4932709, -198.6312103, 259.5796204, -447.1396484, 442.1243591
2: -187.7182159, 247.8797760, -198.8130798, 264.4493713, -452.1676025, 446.6928101
3: -220.0472717, 280.7072754, -233.6219635, 299.2090454, -519.2563477, 514.3292236
4: -189.6266632, 285.1177368, -201.1061249, 303.9932556, -493.6199036, 486.2238770

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8512354, upper bound: 398.8882437
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8710937, upper bound: 398.8897633
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -181.0280762, 291.8794556, -171.1241913, 273.7840576, -454.8121033, 463.0036621
1: -198.6312103, 259.5796204, -187.5600433, 243.4932709, -442.1243591, 447.1396484
2: -198.8130798, 264.4493713, -187.7182159, 247.8797760, -446.6928101, 452.1676025
3: -233.6219635, 299.2090454, -220.0472717, 280.7072754, -514.3292236, 519.2563477
4: -201.1061249, 303.9932556, -189.6266632, 285.1177368, -486.2238770, 493.6199036

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8892716, upper bound: 398.8652212
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8882486, upper bound: 398.8646369
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -181.0280762, 291.8794556, -181.0280762, 291.8794556, -472.9075317, 472.9075317
1: -198.6312103, 259.5796204, -198.6312103, 259.5796204, -458.2107849, 458.2107849
2: -198.8130798, 264.4493713, -198.8130798, 264.4493713, -463.2624207, 463.2624207
3: -233.6219635, 299.2090454, -233.6219635, 299.2090454, -532.8309326, 532.8309937
4: -201.1061249, 303.9932556, -201.1061249, 303.9932556, -505.0993652, 505.0993652

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8892716, upper bound: 398.8886066
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8882486, upper bound: 398.8885849
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -368.1796265, 551.0806885, -695.0838013, 598.7336426
1: -164.6911926, 217.2070312, -402.1123962, 504.7675476, -663.0051880, 606.8721924
2: -164.6050262, 220.8984070, -401.0971069, 513.7360229, -671.9145508, 611.5249023
3: -193.9470978, 250.2816925, -468.4863281, 582.7603149, -769.0442505, 709.0499268
4: -167.2314453, 253.8219147, -401.7456055, 588.3703003, -752.1649170, 646.2683105

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8766690, upper bound: 398.8634257
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881671, upper bound: 398.8648875
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -149.8447571, 244.7119904, -377.3660583, 567.8796997, -712.9656372, 608.1686401
1: -164.6911926, 217.2070312, -412.4937744, 519.7845459, -678.8864136, 617.4164429
2: -164.6050262, 220.8984070, -411.3851013, 528.9593506, -688.1504517, 622.0609741
3: -193.9470978, 250.2816925, -481.2438049, 599.9193726, -787.2591553, 721.9576416
4: -167.2314453, 253.8219147, -412.7245483, 605.9509277, -770.4213867, 657.5633545

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8766690, upper bound: 398.8803114
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881671, upper bound: 398.8819861
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -368.5884399, 551.6654053, -705.1576538, 618.2774658
1: -175.3673859, 233.4939575, -402.5755615, 505.3256836, -674.4139404, 624.0594482
2: -175.2652588, 237.8358307, -401.5458374, 514.2865601, -683.0624390, 629.4147339
3: -207.2741852, 269.0414734, -469.0303040, 583.4158325, -783.2224731, 728.8292847
4: -178.3538513, 272.8920898, -402.2010193, 589.0086670, -764.2554321, 666.2577515

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8976765, upper bound: 398.8635906
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9123302, upper bound: 398.8650324
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -159.3470764, 263.1299133, -377.7867737, 568.4724731, -723.1356812, 627.8597412
1: -175.3673859, 233.4939575, -412.9711609, 520.3518066, -690.3804932, 634.7256470
2: -175.2652588, 237.8358307, -411.8473206, 529.5188599, -699.3721313, 640.0911255
3: -207.2741852, 269.0414734, -481.8051758, 600.5861206, -801.5375366, 741.8847656
4: -178.3538513, 272.8920898, -413.1943054, 606.6036987, -782.6116943, 677.6502075

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8976765, upper bound: 398.8799661
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9123303, upper bound: 398.8817628
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -145.9805450, 238.6965179, -389.1394653, 582.7466431, -721.4857178, 610.7688599
1: -160.3719025, 211.6016083, -424.9147949, 533.7556763, -686.1155396, 621.2728271
2: -160.3276825, 215.2067719, -423.9130249, 543.5217896, -696.0635376, 626.0274048
3: -188.8377380, 243.8211670, -495.0856323, 616.3356934, -795.8067017, 727.0442505
4: -162.8201447, 247.2887573, -424.2561951, 622.2289429, -780.5590210, 660.3650513

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8840565, upper bound: 398.8781047
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8839496, upper bound: 398.8789833
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8843622, upper bound: 398.8789767
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -148.2115021, 241.7634125, -397.1396790, 595.8652344, -737.2896729, 622.7670288
1: -162.8761597, 214.5066833, -433.9001160, 545.8204346, -701.1542969, 634.0517578
2: -162.7887878, 218.1775208, -432.8012085, 555.6649780, -711.1655884, 638.8194580
3: -191.7570038, 247.1793823, -505.7801514, 630.1279297, -812.9664917, 741.8229980
4: -165.3432465, 250.7511749, -433.5382996, 636.2948608, -797.6264648, 673.8168335

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8595989, upper bound: 398.8633966
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8577314, upper bound: 398.8634688
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -155.5301819, 257.2647095, -391.8215332, 586.8251953, -735.1450806, 632.8133545
1: -171.1094055, 228.0290833, -427.8508911, 537.5418091, -700.8314819, 641.1289673
2: -171.0468140, 232.2736053, -426.8645020, 547.3568115, -710.5416260, 646.6010132
3: -202.2516327, 262.7688904, -498.5326843, 620.7286377, -813.8029175, 750.0642700
4: -174.0459595, 266.5260925, -427.1919250, 626.5961914, -796.5262451, 683.0273438

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9052002, upper bound: 398.8787053
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9060925, upper bound: 398.8788472
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -157.5891876, 260.0910339, -399.8564453, 599.9146118, -750.7130127, 644.5018921
1: -173.4123535, 230.7054138, -436.8658142, 549.5973511, -715.6250000, 653.6188354
2: -173.3084412, 235.0266571, -435.7799988, 559.4945068, -725.4022827, 659.1380615
3: -204.9379272, 265.8355408, -509.2384949, 634.5085449, -830.6886597, 764.3837280
4: -176.3238831, 269.6820374, -436.4888000, 640.6397705, -813.3025513, 696.0973511

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9026926, upper bound: 398.8645914
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9048110, upper bound: 398.8664031
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -166.5733948, 266.7813416, -354.1828918, 529.0367432, -688.9498291, 605.7133789
1: -182.5161591, 237.0055695, -386.5718689, 484.1250916, -659.3994751, 610.7081299
2: -182.6948242, 241.2886963, -385.6394043, 492.9137268, -668.3645630, 615.9155273
3: -214.1276398, 273.2288513, -450.0582275, 559.1230469, -764.8930054, 713.2829590
4: -184.4762421, 277.5671692, -385.8050537, 564.4237061, -744.7886353, 653.7434082

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8645081, upper bound: 398.8575031
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651774, upper bound: 398.8584290
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8584290
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8584290
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -166.5733948, 266.7813416, -363.8482971, 546.4477539, -707.5358887, 615.6469116
1: -182.5161591, 237.0055695, -397.4931946, 499.7039185, -675.9282837, 621.8109741
2: -182.6948242, 241.2886963, -396.4274902, 508.6898499, -685.2385864, 626.9674072
3: -214.1276398, 273.2288513, -463.4666748, 576.9177856, -783.8447266, 726.8735962
4: -184.4762421, 277.5671692, -397.3281860, 582.7031860, -763.7823486, 665.6260376

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8645081, upper bound: 398.8769406
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649841, upper bound: 398.8778737
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8778737
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8778737
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -176.6123657, 285.1778259, -354.7385864, 529.9677734, -699.7984009, 625.6303711
1: -193.7355957, 253.3409424, -387.1957092, 484.9654236, -671.4673462, 628.3212280
2: -193.9372253, 258.1333923, -386.2647095, 493.7600098, -680.4094238, 634.1383667
3: -227.8819580, 292.0210571, -450.8115540, 560.1094360, -779.7032471, 733.5914917
4: -196.1167755, 296.7211914, -386.4385071, 565.4050903, -757.6547852, 673.9192505

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8885357, upper bound: 398.8434518
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8880082, upper bound: 398.8585526
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -176.6123657, 285.1778259, -364.8054504, 548.0599365, -719.1532593, 636.1007690
1: -193.7355957, 253.3409424, -398.5668030, 501.1609802, -688.6808472, 639.9778442
2: -193.9372253, 258.1333923, -397.5095215, 510.1612549, -697.9771729, 645.7690430
3: -227.8819580, 292.0210571, -464.7731934, 578.6191406, -799.4503174, 747.8623047
4: -196.1167755, 296.7211914, -398.4286194, 584.4172363, -777.4586792, 686.3468628

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8885357, upper bound: 398.8771915
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8880083, upper bound: 398.8772736
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -167.3997192, 267.2848816, -370.2761536, 556.7968750, -719.1912231, 623.4005127
1: -183.4465790, 237.6695557, -404.8019409, 509.7077026, -687.3488770, 630.6295166
2: -183.5773926, 242.1056824, -403.5724182, 518.7369385, -696.6483765, 635.7413330
3: -215.1263275, 274.0444641, -472.2830505, 588.3764038, -796.7644043, 737.2928467
4: -185.4245758, 278.3859558, -404.9936218, 594.2445679, -776.7622070, 674.8030396

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

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
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600712, upper bound: 398.8646031
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600712, upper bound: 398.8646031
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -169.0650330, 270.1494141, -374.2991638, 564.1031494, -728.3429565, 630.4467773
1: -185.2740173, 240.1738281, -409.2660217, 515.7642212, -695.3597412, 637.7504883
2: -185.4250641, 244.6208496, -408.0673218, 524.8836670, -704.8126831, 642.8950195
3: -217.3108978, 276.9140625, -477.5811768, 595.2750244, -805.9682617, 745.5413818
4: -187.2908783, 281.3237305, -409.5409546, 601.3512573, -785.9022217, 682.3535767

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

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
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8630903, upper bound: 398.8662326
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8630903, upper bound: 398.8662326
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -177.3928223, 285.6380615, -370.8340454, 557.7373657, -730.0194702, 643.3539429
1: -194.6231995, 253.9866791, -405.4295959, 510.5574036, -699.4641724, 648.2858276
2: -194.7671661, 258.8013306, -404.1925049, 519.5916748, -708.6528320, 654.0334473
3: -228.8549805, 292.7921448, -473.0439148, 589.3597412, -811.6403198, 757.6304321
4: -196.9976196, 297.4962158, -405.6348267, 595.2334595, -789.6392822, 695.0855103

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8868516, upper bound: 398.8646960
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8868516, upper bound: 398.8646960
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -178.7953339, 287.9970703, -374.8203125, 564.9931030, -738.8779907, 649.8835449
1: -196.1497040, 256.0394287, -409.8516235, 516.5646362, -707.1090698, 654.9188843
2: -196.3333435, 260.8535767, -408.6472168, 525.6901855, -716.5217285, 660.6711426
3: -230.6506042, 295.1493835, -478.2915344, 596.2000122, -820.3854980, 765.3237915
4: -198.5628510, 299.9086304, -410.1404724, 602.2837524, -798.4114990, 702.0256348

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870890, upper bound: 398.8664066
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870890, upper bound: 398.8664066
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -368.1796265, 551.0806885, -149.8447571, 244.7119904, -598.7336426, 695.0838013
1: -402.1123962, 504.7675476, -164.6911926, 217.2070312, -606.8721924, 663.0052490
2: -401.0971069, 513.7360229, -164.6050262, 220.8984070, -611.5249634, 671.9145508
3: -468.4863281, 582.7603149, -193.9470978, 250.2816925, -709.0499268, 769.0442505
4: -401.7456055, 588.3703003, -167.2314453, 253.8219147, -646.2683716, 752.1649170

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8570725, upper bound: 398.8852537
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8620268, upper bound: 398.8858602
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -368.5884399, 551.6654053, -159.3470764, 263.1299133, -618.2774658, 705.1575928
1: -402.5755615, 505.3256836, -175.3673859, 233.4939575, -624.0594482, 674.4139404
2: -401.5458374, 514.2865601, -175.2652588, 237.8358307, -629.4147339, 683.0624390
3: -469.0303040, 583.4158325, -207.2741852, 269.0414734, -728.8292847, 783.2224121
4: -402.2010193, 589.0086670, -178.3538513, 272.8920898, -666.2577515, 764.2554321

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8570725, upper bound: 398.9112185
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8620268, upper bound: 398.9111841
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -377.3660278, 567.8796387, -149.8447571, 244.7119904, -608.1685791, 712.9655151
1: -412.4937744, 519.7845459, -164.6911926, 217.2070312, -617.4163208, 678.8863525
2: -411.3850403, 528.9592896, -164.6050262, 220.8984070, -622.0609741, 688.1503296
3: -481.2438049, 599.9193726, -193.9470978, 250.2816925, -721.9576416, 787.2590942
4: -412.7245483, 605.9508667, -167.2314453, 253.8219147, -657.5633545, 770.4213257

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792022, upper bound: 398.8852722
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817835, upper bound: 398.8861650
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -377.7867737, 568.4724731, -159.3470764, 263.1299133, -627.8596802, 723.1357422
1: -412.9711609, 520.3518066, -175.3673859, 233.4939575, -634.7257080, 690.3804932
2: -411.8473206, 529.5188599, -175.2652588, 237.8358307, -640.0911255, 699.3721313
3: -481.8051758, 600.5861206, -207.2741852, 269.0414734, -741.8847656, 801.5375366
4: -413.1943054, 606.6036987, -178.3538513, 272.8920898, -677.6502075, 782.6116943

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792022, upper bound: 398.9108144
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817835, upper bound: 398.9116929
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -368.6364441, 551.6452637, -171.1241913, 273.7840576, -627.8082275, 716.6652222
1: -402.6218872, 505.3193665, -187.5600433, 243.4932709, -633.9542847, 686.1938477
2: -401.5912781, 514.2757568, -187.7182159, 247.8797760, -639.1376953, 695.3139648
3: -469.0668335, 583.4095459, -220.0472717, 280.7072754, -740.4062500, 795.6820679
4: -402.2324829, 588.9990845, -189.6266632, 285.1177368, -678.2562866, 774.9708862

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8568216, upper bound: 398.8670717
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8619825, upper bound: 398.8695184
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -368.9359131, 552.1428833, -181.0280762, 291.8794556, -647.1451416, 726.8825073
1: -402.9606628, 505.7763672, -198.6312103, 259.5796204, -651.0089111, 697.7249756
2: -401.9247742, 514.7333374, -198.8130798, 264.4493713, -656.7949829, 706.7469482
3: -469.4751892, 583.9439697, -233.6219635, 299.2090454, -760.0349121, 809.8460083
4: -402.5740967, 589.5267944, -201.1061249, 303.9932556, -697.8855591, 787.1943359

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8568216, upper bound: 398.8870135
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8619825, upper bound: 398.8866890
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -377.6723633, 568.2247314, -171.1241913, 273.7840576, -637.0986328, 734.3327637
1: -412.8417969, 520.1389771, -187.5600433, 243.4932709, -644.3417969, 701.8820190
2: -411.7131042, 529.3001709, -187.7182159, 247.8797760, -649.5136108, 711.3544312
3: -481.6389771, 600.3395996, -220.0472717, 280.7072754, -753.1297607, 813.6737061
4: -413.0537720, 606.3499756, -189.6266632, 285.1177368, -689.3984985, 793.0002441

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8669746
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815648, upper bound: 398.8702704
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -378.0595093, 568.8591309, -181.0280762, 291.8794556, -656.6500244, 744.7622070
1: -413.2799683, 520.7207642, -198.6312103, 259.5796204, -661.5917358, 713.5989990
2: -412.1460571, 529.8829346, -198.8130798, 264.4493713, -667.3849487, 722.9763794
3: -482.1681824, 601.0198364, -233.6219635, 299.2090454, -772.9956665, 828.0602417
4: -413.4971619, 607.0269775, -201.1061249, 303.9932556, -709.2033691, 805.4457397

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8876432
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815649, upper bound: 398.8891845
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -389.1394653, 582.7466431, -145.9805450, 238.6965179, -610.7688599, 721.4857178
1: -424.9147949, 533.7556763, -160.3719025, 211.6016083, -621.2728271, 686.1155396
2: -423.9130249, 543.5217896, -160.3276825, 215.2067719, -626.0274048, 696.0635376
3: -495.0856323, 616.3356934, -188.8377380, 243.8211670, -727.0442505, 795.8067017
4: -424.2561951, 622.2289429, -162.8201447, 247.2887573, -660.3649902, 780.5590820

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8590516, upper bound: 398.8843406
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8590516, upper bound: 398.8843892
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -391.8214722, 586.8251953, -155.5301819, 257.2647095, -632.8133545, 735.1450806
1: -427.8508606, 537.5418091, -171.1094055, 228.0290833, -641.1289673, 700.8314819
2: -426.8645020, 547.3568726, -171.0468140, 232.2736053, -646.6010132, 710.5415649
3: -498.5326538, 620.7286987, -202.2516327, 262.7688904, -750.0643311, 813.8029785
4: -427.1919250, 626.5961914, -174.0459595, 266.5260925, -683.0273438, 796.5262451

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8591439, upper bound: 398.9061460
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8591439, upper bound: 398.9062117
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -397.2176514, 595.9394531, -148.1897583, 241.7347412, -622.8064575, 737.3425293
1: -433.9887390, 545.9011841, -162.8532562, 214.4815521, -634.1040039, 701.2117310
2: -432.8850708, 555.7412720, -162.7655182, 218.1510315, -638.8685913, 711.2190552
3: -505.8800354, 630.2239990, -191.7318115, 247.1504364, -741.8830566, 813.0369873
4: -433.6221313, 636.3857422, -165.3212433, 250.7205963, -673.8638916, 797.6953125

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8645531, upper bound: 398.8737902
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8662317, upper bound: 398.8784625
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -399.9309082, 599.9860229, -157.5772552, 260.0753784, -644.5515747, 750.7726440
1: -436.9504089, 549.6751099, -173.3998566, 230.6916504, -653.6788330, 715.6900635
2: -435.8601685, 559.5680542, -173.2957306, 235.0122375, -659.1958008, 725.4633179
3: -509.3338318, 634.6009521, -204.9243011, 265.8196106, -764.4528809, 830.7671509
4: -436.5688477, 640.7274780, -176.3119202, 269.6653748, -696.1547852, 813.3781738

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8647294, upper bound: 398.9044076
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8664031, upper bound: 398.9048110
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -391.2413635, 585.7997437, -166.5733948, 266.7813416, -640.3890991, 744.6883545
1: -427.1936340, 536.6293945, -182.5161591, 237.0055695, -649.1060181, 710.7780151
2: -426.2079163, 546.4346924, -182.6948242, 241.2886963, -654.4146118, 720.9298096
3: -497.7249146, 619.6704712, -214.1276398, 273.2288513, -759.2951660, 824.1715088
4: -426.5057068, 625.5219116, -184.4762421, 277.5671692, -693.0066528, 805.1965332

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8584290, upper bound: 398.8652116
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8187013, upper bound: 398.8637153
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8778143, upper bound: 398.8647627
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -392.2433777, 587.4143066, -176.6123657, 285.1778259, -660.8732910, 756.3272095
1: -428.3204041, 538.1027832, -193.7355957, 253.3409424, -667.3187866, 723.5687866
2: -427.3262024, 547.9130249, -193.9372253, 258.1333923, -673.2399902, 733.6976929
3: -499.0840149, 621.3830566, -227.8819580, 292.0210571, -780.3316650, 839.8169556
4: -427.6523743, 627.2429810, -196.1167755, 296.7211914, -713.7755127, 818.8991699

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8586497, upper bound: 398.8890656
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8586497, upper bound: 398.8891087
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -399.8821716, 599.7673340, -169.6011047, 270.9648438, -654.1116943, 762.1998901
1: -436.8732300, 549.4862061, -185.8597412, 240.9204102, -663.6195679, 727.4682617
2: -435.7966309, 559.3796387, -186.0190887, 245.3761444, -668.9414673, 737.7204590
3: -509.2102051, 634.3873901, -217.9990540, 277.7748108, -776.1062622, 843.2266846
4: -436.4600220, 640.5150757, -187.8750916, 282.1921692, -708.2759399, 824.0921631

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646031, upper bound: 398.8600712
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8662326, upper bound: 398.8630903
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -400.5055542, 600.7150879, -179.5081482, 289.1158142, -673.9287720, 772.9546509
1: -437.5761108, 550.3670044, -196.9323730, 257.0628357, -681.1687012, 739.4979858
2: -436.4825439, 560.2562866, -197.1161957, 261.8901062, -687.0911255, 749.6710815
3: -510.0463562, 635.4094849, -231.5811310, 296.3304138, -796.3208618, 857.9821777
4: -437.1647339, 641.5315552, -199.3473969, 301.1004028, -728.3769531, 836.8851929

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646960, upper bound: 398.8868516
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8664066, upper bound: 398.8870890
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -369.0663452, 552.3657227, -369.0663452, 552.3657227, -897.5636597, 897.5636597
1: -403.1070862, 505.9794922, -403.1070862, 505.9794922, -886.0205688, 886.0205078
2: -402.0683594, 514.9389648, -402.0683594, 514.9389648, -895.3437500, 895.3437500
3: -469.6511230, 584.1793823, -469.6511230, 584.1793823, -1032.6239014, 1032.6240234
4: -402.7211609, 589.7617798, -402.7211609, 589.7617798, -976.2642212, 976.2642212

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8565962, upper bound: 398.8608000
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8613514, upper bound: 398.8613514
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -369.0663452, 552.3657227, -378.1610107, 569.0414429, -915.3328247, 906.9153442
1: -403.1070862, 505.9794922, -413.3947754, 520.8857422, -901.7987671, 896.4772339
2: -402.0683594, 514.9389648, -412.2584534, 530.0501709, -911.4743042, 905.7894897
3: -469.6511230, 584.1793823, -482.3082886, 601.2113037, -1050.7216797, 1045.4322510
4: -402.7211609, 589.7617798, -413.6139221, 607.2186279, -994.4020386, 987.4796753

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8565962, upper bound: 398.8808416
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8613514, upper bound: 398.8809811
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -378.1610107, 569.0414429, -369.0663452, 552.3657227, -906.9153442, 915.3327637
1: -413.3947754, 520.8857422, -403.1070862, 505.9794922, -896.4772949, 901.7987671
2: -412.2584534, 530.0501709, -402.0683594, 514.9389648, -905.7895508, 911.4743042
3: -482.3082886, 601.2113037, -469.6511230, 584.1793823, -1045.4322510, 1050.7215576
4: -413.6139221, 607.2186279, -402.7211609, 589.7617798, -987.4796753, 994.4020386

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788547, upper bound: 398.8608192
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809811, upper bound: 398.8617019
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -378.1610107, 569.0414429, -378.1610107, 569.0414429, -924.7573853, 924.7573242
1: -413.3947754, 520.8857422, -413.3947754, 520.8857422, -912.3007202, 912.3007202
2: -412.2584534, 530.0501709, -412.2584534, 530.0501709, -921.9827271, 921.9827271
3: -482.3082886, 601.2113037, -482.3082886, 601.2113037, -1063.5869141, 1063.5869141
4: -413.6139221, 607.2186279, -413.6139221, 607.2186279, -1005.6649780, 1005.6649170

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8788547, upper bound: 398.8803802
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809811, upper bound: 398.8812538
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -369.0663452, 552.3657227, -393.9781494, 588.2157593, -931.3281250, 919.9601440
1: -403.1070862, 505.9794922, -430.0381470, 539.1370850, -917.1663818, 910.6619263
2: -402.0683594, 514.9389648, -429.1344604, 548.8555298, -927.2760620, 920.1653442
3: -469.6511230, 584.1793823, -500.5832214, 622.5844727, -1068.6796875, 1061.7585449
4: -402.7211609, 589.7617798, -428.9787292, 628.2936401, -1013.4653931, 1000.7791748

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8567874, upper bound: 398.8621422
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8619174, upper bound: 398.8642747
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -369.0663452, 552.3657227, -403.0519409, 604.5369263, -948.7670288, 929.3117065
1: -403.1070862, 505.9794922, -440.3587952, 553.8668213, -932.7525024, 921.1540527
2: -402.0683594, 514.9389648, -439.2883301, 563.7449951, -943.1919556, 930.5952759
3: -469.6511230, 584.1793823, -513.2688599, 639.4283447, -1086.5598145, 1074.6229248
4: -402.7211609, 589.7617798, -439.9255066, 645.6051025, -1031.4665527, 1012.0736694

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8567874, upper bound: 398.8808300
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8619174, upper bound: 398.8809217
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -378.1610107, 569.0414429, -393.9781494, 588.2157593, -940.6798706, 937.7294312
1: -413.3947754, 520.8857422, -430.0381470, 539.1370850, -927.6229858, 926.4401855
2: -412.2584534, 530.0501709, -429.1344604, 548.8555298, -937.7219238, 936.2958984
3: -482.3082886, 601.2113037, -500.5832214, 622.5844727, -1081.4880371, 1079.8562012
4: -413.6139221, 607.2186279, -428.9787292, 628.2936401, -1024.6809082, 1018.9170532

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8619663
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814801, upper bound: 398.8646166
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -378.1610107, 569.0414429, -403.0519409, 604.5369263, -958.1871338, 947.1536865
1: -413.3947754, 520.8857422, -440.3587952, 553.8668213, -943.2503662, 936.9774170
2: -412.2584534, 530.0501709, -439.2883301, 563.7449951, -953.6964722, 946.7885132
3: -482.3082886, 601.2113037, -513.2688599, 639.4283447, -1099.4194336, 1092.7785645
4: -413.6139221, 607.2186279, -439.9255066, 645.6051025, -1042.7257080, 1030.2583008

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8803297
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814801, upper bound: 398.8812437
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -390.9489746, 584.9989014, -361.9461975, 543.4219360, -908.2990723, 921.4278564
1: -426.8886108, 536.0995483, -395.4551086, 497.1731873, -898.7390747, 906.6184082
2: -425.8767090, 545.8541260, -394.3397217, 506.1228027, -908.3939819, 917.0237427
3: -497.3632812, 619.1048584, -461.1635742, 574.0758057, -1048.8302002, 1057.1506348
4: -426.1755371, 624.8766479, -395.3106384, 579.7074585, -988.2597656, 1003.0172119

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800611, upper bound: 398.8814948
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802030, upper bound: 398.8816772
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -392.2724304, 587.3761597, -365.3819580, 549.6018066, -916.1138916, 927.5167236
1: -428.3405151, 538.0778809, -399.2607117, 502.2415771, -905.5059204, 912.6513672
2: -427.3552856, 547.8843994, -398.1781616, 511.2810974, -915.3065796, 923.1404419
3: -499.0904236, 621.3626099, -465.6717834, 579.8666382, -1056.5452881, 1064.1169434
4: -427.6501770, 627.2144165, -399.1698303, 585.7011719, -995.9699097, 1009.3717041

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811986, upper bound: 398.8815982
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814531, upper bound: 398.8817321
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -399.4141235, 598.3980713, -361.9461975, 543.4219360, -917.0098877, 934.8962402
1: -436.3414612, 548.4771729, -395.4551086, 497.1731873, -908.3630981, 918.9964600
2: -435.2403870, 558.3145752, -394.3397217, 506.1228027, -917.9523926, 929.4807739
3: -508.5097961, 633.2675171, -461.1635742, 574.0758057, -1060.0651855, 1071.3322754
4: -435.8445435, 639.3053589, -395.3106384, 579.7074585, -997.9844360, 1017.4420776

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649711, upper bound: 398.8739456
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649711, upper bound: 398.8739456
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -400.7750244, 600.9243164, -365.3819580, 549.6018066, -924.8723755, 941.1565552
1: -437.8368225, 550.5580444, -399.2607117, 502.2415771, -915.1869507, 925.1524658
2: -436.7642517, 560.4452515, -398.1781616, 511.2810974, -924.9191284, 935.7128296
3: -510.3007202, 635.6403198, -465.6717834, 579.8666382, -1067.8620605, 1078.4362793
4: -437.3742676, 641.7625122, -399.1698303, 585.7011719, -1005.7597046, 1023.9323730

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8691640, upper bound: 398.8788475
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8662451, upper bound: 398.8781524
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8662451, upper bound: 398.8788476
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -390.7001648, 584.5028687, -371.5879822, 558.6466675, -923.3612061, 930.7387085
1: -426.6116028, 535.6382446, -406.2244873, 511.3952637, -912.7219849, 917.0441284
2: -425.5963440, 545.3900146, -405.0018311, 520.4406128, -922.4616089, 927.3385620
3: -497.0162354, 618.5798340, -473.9243469, 590.3355713, -1064.8076172, 1069.4420166
4: -425.8708801, 624.3669434, -406.3762817, 596.2328491, -1004.5034180, 1013.6392822

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8610152, upper bound: 398.8648884
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8765894, upper bound: 398.8649682
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -392.0235596, 586.8800659, -375.5977478, 565.9308472, -932.3377686, 937.4307861
1: -428.0636597, 537.6168213, -410.6718750, 517.4291382, -920.5077515, 923.7681885
2: -427.0749207, 547.4200439, -409.4814453, 526.5659180, -930.3934326, 934.1208496
3: -498.7434998, 620.8380737, -479.2002869, 597.2065430, -1073.6644287, 1077.2093506
4: -427.3457031, 626.7049561, -410.9074707, 603.3143311, -1013.3374634, 1020.6872559

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8146993, upper bound: 398.8689265
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785341, upper bound: 398.8696071
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -399.4141235, 598.3980713, -371.5879822, 558.6466675, -933.0400391, 945.4067383
1: -436.3414612, 548.4771729, -406.2244873, 511.3952637, -923.3765259, 930.6271973
2: -435.2403870, 558.3145752, -405.0018311, 520.4406128, -933.0919189, 941.0413208
3: -508.5097961, 633.2675171, -473.9243469, 590.3355713, -1077.0745850, 1084.8333740
4: -435.8445435, 639.3053589, -406.3762817, 596.2328491, -1015.2236328, 1029.2380371

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649633, upper bound: 398.8651942
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649633, upper bound: 398.8656553
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -400.7750244, 600.9243164, -375.5977478, 565.9308472, -942.0445557, 952.2478638
1: -437.8368225, 550.5580444, -410.6718750, 517.4291382, -931.1992798, 937.4521484
2: -436.7642517, 560.4452515, -409.4814453, 526.5659180, -941.0596924, 947.9165649
3: -510.3007202, 635.6403198, -479.2002869, 597.2065430, -1085.9929199, 1092.7170410
4: -437.3742676, 641.7625122, -410.9074707, 603.3143311, -1024.1079102, 1036.4033203

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8656553, upper bound: 398.8675457
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8656553, upper bound: 398.8680063
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.87 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8780922, upper bound: 398.8910622
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8971743, upper bound: 398.8971743
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8780922, upper bound: 398.9159720
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8971743, upper bound: 398.9239766
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8994488, upper bound: 398.8911733
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9239765, upper bound: 398.8974361
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8994489, upper bound: 398.9158200
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9239765, upper bound: 398.9239204
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8725274, upper bound: 398.8650562
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8919743, upper bound: 398.8698238
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8725274, upper bound: 398.8807459
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8919743, upper bound: 398.8891460
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8904131, upper bound: 398.8653017
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9154555, upper bound: 398.8699556
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8904131, upper bound: 398.8810251
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9154555, upper bound: 398.8903988
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8516072, upper bound: 398.8886582
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8516072, upper bound: 398.8933591
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8516072, upper bound: 398.9142135
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8711674, upper bound: 398.9163761
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8893602, upper bound: 398.8893467
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8882486, upper bound: 398.8893518
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8893603, upper bound: 398.9142401
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8882486, upper bound: 398.9140139
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8512354, upper bound: 398.8656831
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8710937, upper bound: 398.8710971
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8512354, upper bound: 398.8882437
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8710937, upper bound: 398.8897633
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8892716, upper bound: 398.8652212
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8882486, upper bound: 398.8646369
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8892716, upper bound: 398.8886066
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8882486, upper bound: 398.8885849
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8766690, upper bound: 398.8634257
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8881671, upper bound: 398.8648875
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8766690, upper bound: 398.8803114
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8881671, upper bound: 398.8819861
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8976765, upper bound: 398.8635906
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9123302, upper bound: 398.8650324
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8976765, upper bound: 398.8799661
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9123303, upper bound: 398.8817628
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8839496, upper bound: 398.8789833
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8843622, upper bound: 398.8789767
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8595989, upper bound: 398.8633966
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8577314, upper bound: 398.8634688
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9052002, upper bound: 398.8787053
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9060925, upper bound: 398.8788472
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9026926, upper bound: 398.8645914
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.9048110, upper bound: 398.8664031
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8584290
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8584290
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8778737
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8652116, upper bound: 398.8778737
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8885357, upper bound: 398.8434518
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8880082, upper bound: 398.8585526
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8885357, upper bound: 398.8771915
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8880083, upper bound: 398.8772736
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8600712, upper bound: 398.8646031
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8600712, upper bound: 398.8646031
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8630903, upper bound: 398.8662326
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8630903, upper bound: 398.8662326
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8868516, upper bound: 398.8646960
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8868516, upper bound: 398.8646960
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8870890, upper bound: 398.8664066
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8870890, upper bound: 398.8664066
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8570725, upper bound: 398.8852537
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8620268, upper bound: 398.8858602
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8570725, upper bound: 398.9112185
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8620268, upper bound: 398.9111841
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8792022, upper bound: 398.8852722
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8817835, upper bound: 398.8861650
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8792022, upper bound: 398.9108144
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8817835, upper bound: 398.9116929
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8568216, upper bound: 398.8670717
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8619825, upper bound: 398.8695184
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8568216, upper bound: 398.8870135
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8619825, upper bound: 398.8866890
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8669746
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8815648, upper bound: 398.8702704
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8876432
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8815649, upper bound: 398.8891845
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8590516, upper bound: 398.8843406
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8590516, upper bound: 398.8843892
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8591439, upper bound: 398.9061460
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8591439, upper bound: 398.9062117
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8645531, upper bound: 398.8737902
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8662317, upper bound: 398.8784625
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8647294, upper bound: 398.9044076
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8664031, upper bound: 398.9048110
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8187013, upper bound: 398.8637153
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8778143, upper bound: 398.8647627
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8586497, upper bound: 398.8890656
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8586497, upper bound: 398.8891087
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8646031, upper bound: 398.8600712
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8662326, upper bound: 398.8630903
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8646960, upper bound: 398.8868516
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8664066, upper bound: 398.8870890
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8565962, upper bound: 398.8608000
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8613514, upper bound: 398.8613514
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8565962, upper bound: 398.8808416
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8613514, upper bound: 398.8809811
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8788547, upper bound: 398.8608192
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8809811, upper bound: 398.8617019
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8788547, upper bound: 398.8803802
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8809811, upper bound: 398.8812538
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8567874, upper bound: 398.8621422
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8619174, upper bound: 398.8642747
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8567874, upper bound: 398.8808300
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8619174, upper bound: 398.8809217
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8619663
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8814801, upper bound: 398.8646166
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8787594, upper bound: 398.8803297
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8814801, upper bound: 398.8812437
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8800611, upper bound: 398.8814948
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8802030, upper bound: 398.8816772
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8811986, upper bound: 398.8815982
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8814531, upper bound: 398.8817321
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8649711, upper bound: 398.8739456
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8649711, upper bound: 398.8739456
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8662451, upper bound: 398.8781524
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8662451, upper bound: 398.8788476
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8610152, upper bound: 398.8648884
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8765894, upper bound: 398.8649682
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8146993, upper bound: 398.8689265
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8785341, upper bound: 398.8696071
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8649633, upper bound: 398.8651942
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8649633, upper bound: 398.8656553
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8656553, upper bound: 398.8675457
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.87
Output dim: 0, lower bound: -398.8656553, upper bound: 398.8680063

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -155.1507874, 248.9590302, -146.4665985, 239.3150635, -394.4658508, 395.4256287
1: -170.2216187, 221.1958771, -160.9658966, 212.3166504, -382.5382690, 382.1617737
2: -170.0487823, 225.1402435, -160.8631897, 215.9369507, -385.9856873, 386.0033875
3: -199.5221863, 254.9492798, -189.4866180, 244.6451111, -444.1672974, 444.4359131
4: -171.9838409, 258.9958496, -163.4032135, 248.1358643, -420.1196899, 422.3990479

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8780141, upper bound: 398.8780141
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8780141, upper bound: 398.8910622
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -147.7432098, 240.7355347, -149.8447571, 244.7119904, -392.4551086, 390.5802917
1: -162.3699646, 213.7429810, -164.6911926, 217.2070312, -379.5769958, 378.4341736
2: -162.2663574, 217.4007111, -164.6050262, 220.8984070, -383.1647644, 382.0057373
3: -191.1653748, 246.2875519, -193.9470978, 250.2816925, -441.4470215, 440.2346497
4: -164.8296509, 249.8734436, -167.2314453, 253.8219147, -418.6515503, 417.1048889

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8910622, upper bound: 398.8780922
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8910622, upper bound: 398.8971743
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -155.1507874, 248.9590302, -155.9761963, 257.6968384, -412.8476257, 404.9352417
1: -170.2216187, 221.1958771, -171.6362762, 228.5842743, -398.8058472, 392.8320923
2: -170.0487823, 225.1402435, -171.5304260, 232.7923584, -402.8410339, 396.6706543
3: -199.5221863, 254.9492798, -202.7875519, 263.4050598, -462.9272461, 457.7368164
4: -171.9838409, 258.9958496, -174.5700836, 267.1423645, -439.1262207, 433.5659180

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8780512, upper bound: 398.8994488
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8780512, upper bound: 398.9159720
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -147.7432098, 240.7355347, -159.3470764, 263.1299133, -410.8731079, 400.0826111
1: -162.3699646, 213.7429810, -175.3673859, 233.4939575, -395.8638916, 389.1103516
2: -162.2663574, 217.4007111, -175.2652588, 237.8358307, -400.1021118, 392.6659546
3: -191.1653748, 246.2875519, -207.2741852, 269.0414734, -460.2068481, 453.5617371
4: -164.8296509, 249.8734436, -178.3538513, 272.8920898, -437.7217407, 428.2272949

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8911733, upper bound: 398.8994488
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8911733, upper bound: 398.9239765
time: 1.02 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.23 + 417.58 = 420.82 seconds
