## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 141.076127489203


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367)
1: (-348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983)
2: (-187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022)
3: (-321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637)
4: (-236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.96 + 2.04 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -141.0803599, upper bound: 141.0803599

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800287, upper bound: 141.0795944
time: 0.79 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -141.0800287, upper bound: 141.0795944
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -87.9863052, 74.2605286, -89.3568115, 75.4509735, -163.4372711, 163.6173401
1: -334.7331543, 276.9394226, -339.8715210, 281.5431824, -616.2761841, 616.8109131
2: -179.8543549, 281.0703125, -182.7203369, 285.6041565, -465.4584961, 463.7905884
3: -308.4378967, 253.0224304, -313.1506348, 257.1592102, -565.5971069, 566.1730347
4: -227.3898468, 283.2803955, -230.8860779, 287.8447876, -515.2346191, 514.1663818

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -98.7185364, 83.2145462, -87.2054443, 73.7391434, -172.4576721, 170.4199829
1: -375.7584839, 310.6868591, -331.1356201, 275.2955017, -651.0539551, 641.8224487
2: -201.3605652, 314.6749573, -178.4256744, 279.3394165, -480.6999207, 493.1006470
3: -345.6825562, 283.6244202, -305.0807800, 251.3557739, -597.0381470, 588.7051392
4: -254.7154388, 316.3996277, -224.7526245, 281.3445129, -536.0598755, 541.1522217

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.44 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -87.9863052, 74.2605286, -87.9863052, 74.2605286, -162.2468262, 162.2468262
1: -334.7331543, 276.9394226, -334.7331543, 276.9394226, -611.6724854, 611.6725464
2: -179.8543549, 281.0703125, -179.8543549, 281.0703125, -460.9246216, 460.9246216
3: -308.4378967, 253.0224304, -308.4378967, 253.0224304, -561.4602661, 561.4602661
4: -227.3898468, 283.2803955, -227.3898468, 283.2803955, -510.6702271, 510.6702271

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0742279, upper bound: 141.0635043
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0628488, upper bound: 141.0628488
time: 0.59 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -87.9863052, 74.2605286, -98.7185364, 83.2145462, -171.2008514, 172.9790649
1: -334.7331543, 276.9394226, -375.7584839, 310.6868591, -645.4197998, 652.6978760
2: -179.8543549, 281.0703125, -201.3605652, 314.6749573, -494.5292969, 482.4307251
3: -308.4378967, 253.0224304, -345.6825562, 283.6244202, -592.0621948, 598.7048340
4: -227.3898468, 283.2803955, -254.7154388, 316.3996277, -543.7894897, 537.9958496

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0742279, upper bound: 141.0795184
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0628488, upper bound: 141.0740116
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -98.7185364, 83.2145462, -87.9863052, 74.2605286, -172.9790649, 171.2008514
1: -375.7584839, 310.6868591, -334.7331543, 276.9394226, -652.6978760, 645.4197998
2: -201.3605652, 314.6749573, -179.8543549, 281.0703125, -482.4307251, 494.5292969
3: -345.6825562, 283.6244202, -308.4378967, 253.0224304, -598.7048340, 592.0621338
4: -254.7154388, 316.3996277, -227.3898468, 283.2803955, -537.9958496, 543.7894897

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794972, upper bound: 141.0795601
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0740115, upper bound: 141.0634861
time: 0.81 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -98.7185364, 83.2145462, -98.7185364, 83.2145462, -181.9330750, 181.9330750
1: -375.7584839, 310.6868591, -375.7584839, 310.6868591, -686.4453125, 686.4453125
2: -201.3605652, 314.6749573, -201.3605652, 314.6749573, -516.0354004, 516.0354614
3: -345.6825562, 283.6244202, -345.6825562, 283.6244202, -629.3068237, 629.3067627
4: -254.7154388, 316.3996277, -254.7154388, 316.3996277, -571.1150513, 571.1150513

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0743358, upper bound: 141.0794908
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0740116, upper bound: 141.0794894
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.43 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0742279, upper bound: 141.0635043
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0628488, upper bound: 141.0628488
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0742279, upper bound: 141.0795184
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0628488, upper bound: 141.0740116
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0794972, upper bound: 141.0795601
NS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0740115, upper bound: 141.0634861
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0743358, upper bound: 141.0794908
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -141.0740116, upper bound: 141.0794894

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -83.4748535, 70.0752945, -95.5780563, 80.3348541, -163.8096924, 165.6533203
1: -317.9730225, 260.8223877, -364.1752930, 299.5302429, -617.5032959, 624.9976196
2: -170.4615173, 265.9446106, -194.9367065, 304.0520020, -474.5134583, 460.8812561
3: -292.9162903, 238.5078583, -334.9004822, 273.6596680, -566.5759277, 573.4083252
4: -215.9780121, 267.3732910, -246.7798920, 305.5356750, -521.5136719, 514.1531372

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799385, upper bound: 141.0795086
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799309, upper bound: 141.0795035
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -95.5780563, 80.3348541, -83.4748535, 70.0752945, -165.6533356, 163.8096924
1: -364.1752930, 299.5302429, -317.9730225, 260.8223877, -624.9976196, 617.5032959
2: -194.9367065, 304.0520020, -170.4615173, 265.9446106, -460.8812561, 474.5134277
3: -334.9004822, 273.6596680, -292.9162903, 238.5078583, -573.4083252, 566.5759277
4: -246.7798920, 305.5356750, -215.9780121, 267.3732910, -514.1531982, 521.5136719

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795086, upper bound: 141.0799385
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795035, upper bound: 141.0799309
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -93.6247330, 78.5752792, -95.5780563, 80.3348541, -173.9595642, 174.1532898
1: -356.9270630, 292.7353516, -364.1752930, 299.5302429, -656.4572754, 656.9106445
2: -190.9016571, 297.5428162, -194.9367065, 304.0520020, -494.9536743, 492.4795227
3: -328.1320496, 267.5841980, -334.9004822, 273.6596680, -601.7916260, 602.4846191
4: -241.7907410, 298.9456787, -246.7798920, 305.5356750, -547.3264160, 545.7254639

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794736, upper bound: 141.0794894
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794736, upper bound: 141.0794894
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.9499969, 103.2073746, -93.6438980, 78.8366623, -201.7865906, 196.8512726
1: -470.1757202, 384.2815857, -356.6582031, 294.2830811, -764.4587402, 740.9398193
2: -248.4368896, 389.9728699, -190.7530365, 297.8670654, -546.3038330, 580.7258911
3: -430.9766846, 350.9409180, -328.2691650, 268.5314636, -699.5081787, 679.2100830
4: -317.0972290, 390.6662903, -241.8630066, 299.5710144, -616.6682129, 632.5292969

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794738, upper bound: 141.0794894
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794738, upper bound: 141.0794894
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.79 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0799385, upper bound: 141.0795086
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0799309, upper bound: 141.0795035
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0795086, upper bound: 141.0799385
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0795035, upper bound: 141.0799309
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0794736, upper bound: 141.0794894
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0794736, upper bound: 141.0794894
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0794738, upper bound: 141.0794894
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.79
Output dim: 0, lower bound: -141.0794738, upper bound: 141.0794894

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -81.9785004, 68.8120193, -93.1954575, 78.3237610, -160.3022614, 162.0074768
1: -312.2781982, 256.0882568, -355.1020813, 292.0031128, -604.2813110, 611.1903076
2: -167.4104309, 261.1761475, -190.0880890, 296.4861450, -463.8965149, 451.2642212
3: -287.7467346, 234.1931763, -326.6278992, 266.7794189, -554.5260010, 560.8208618
4: -212.1408997, 262.6105042, -240.6859283, 297.9858398, -510.1267395, 503.2963562

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799292, upper bound: 141.0795024
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798898, upper bound: 141.0794762
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -82.4576263, 69.2163467, -96.5716095, 81.0334015, -163.4910278, 165.7879639
1: -314.0762939, 257.6392517, -367.7864075, 302.4145813, -616.4907837, 625.4256592
2: -168.4635010, 262.6291199, -197.2401733, 306.0216980, -474.4851990, 459.8692932
3: -289.3600159, 235.5928497, -338.4599304, 276.1612549, -565.5212402, 574.0526733
4: -213.3787079, 264.0035400, -249.7925720, 307.5279236, -520.9066162, 513.7960815

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795156, upper bound: 141.0793985
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0795035
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -93.1954575, 78.3237610, -81.9785004, 68.8120193, -162.0074768, 160.3022614
1: -355.1020813, 292.0031128, -312.2781982, 256.0882568, -611.1903076, 604.2813110
2: -190.0880890, 296.4861450, -167.4104309, 261.1761475, -451.2642212, 463.8965149
3: -326.6278992, 266.7794189, -287.7467346, 234.1931763, -560.8209229, 554.5260010
4: -240.6859283, 297.9858398, -212.1408997, 262.6105042, -503.2963562, 510.1267395

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795024, upper bound: 141.0799292
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794762, upper bound: 141.0798898
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -96.5716095, 81.0334015, -82.4576263, 69.2163467, -165.7879639, 163.4910278
1: -367.7864075, 302.4145813, -314.0762939, 257.6392517, -625.4256592, 616.4907227
2: -197.2401733, 306.0216980, -168.4635010, 262.6291199, -459.8692932, 474.4851990
3: -338.4599304, 276.1612549, -289.3600159, 235.5928497, -574.0526733, 565.5212402
4: -249.7925720, 307.5279236, -213.3787079, 264.0035400, -513.7961426, 520.9066162

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793985, upper bound: 141.0795156
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0799309
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -93.6247330, 78.5752792, -93.6247330, 78.5752792, -172.1999664, 172.1999664
1: -356.9270630, 292.7353516, -356.9270630, 292.7353516, -649.6624146, 649.6624146
2: -190.9016571, 297.5428162, -190.9016571, 297.5428162, -488.4444580, 488.4444580
3: -328.1320496, 267.5841980, -328.1320496, 267.5841980, -595.7161865, 595.7161865
4: -241.7907410, 298.9456787, -241.7907410, 298.9456787, -540.7364502, 540.7364502

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794616, upper bound: 141.0794616
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794619, upper bound: 141.0794783
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -93.6247330, 78.5752792, -122.9499969, 103.2073746, -196.8321075, 201.5252228
1: -356.9270630, 292.7353516, -470.1757202, 384.2815857, -741.2086182, 762.9110718
2: -190.9016571, 297.5428162, -248.4368896, 389.9728699, -580.8745117, 545.9795532
3: -328.1320496, 267.5841980, -430.9766846, 350.9409180, -679.0729370, 698.5608521
4: -241.7907410, 298.9456787, -317.0972290, 390.6662903, -632.4570312, 616.0429077

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794616, upper bound: 141.0794616
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794619, upper bound: 141.0794783
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -122.9499969, 103.2073746, -93.6247330, 78.5752792, -201.5252228, 196.8321075
1: -470.1757202, 384.2815857, -356.9270630, 292.7353516, -762.9110718, 741.2086182
2: -248.4368896, 389.9728699, -190.9016571, 297.5428162, -545.9795532, 580.8745117
3: -430.9766846, 350.9409180, -328.1320496, 267.5841980, -698.5608521, 679.0729370
4: -317.0972290, 390.6662903, -241.7907410, 298.9456787, -616.0429077, 632.4570312

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794616, upper bound: 141.0794780
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794619, upper bound: 141.0794769
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -122.9499969, 103.2073746, -122.9499969, 103.2073746, -226.1573334, 226.1573486
1: -470.1757202, 384.2815857, -470.1757202, 384.2815857, -854.4572754, 854.4572754
2: -248.4368896, 389.9728699, -248.4368896, 389.9728699, -638.4097900, 638.4097900
3: -430.9766846, 350.9409180, -430.9766846, 350.9409180, -781.9176025, 781.9176025
4: -317.0972290, 390.6662903, -317.0972290, 390.6662903, -707.7635498, 707.7635498

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792313, upper bound: 141.0779931
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794738, upper bound: 141.0794888
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.08 seconds
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0798898, upper bound: 141.0794762
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0795035
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794762, upper bound: 141.0798898
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0799309
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794616, upper bound: 141.0794616
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794619, upper bound: 141.0794783
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794616, upper bound: 141.0794616
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794619, upper bound: 141.0794783
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794616, upper bound: 141.0794780
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794619, upper bound: 141.0794769
NS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0792313, upper bound: 141.0779931
NS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 0, lower bound: -141.0794738, upper bound: 141.0794888

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -81.3296356, 68.2649765, -92.4449081, 77.6091919, -158.9388275, 160.7098846
1: -309.7753296, 254.0611267, -352.0039978, 289.2525024, -599.0278320, 606.0651245
2: -166.0993042, 259.0742188, -188.7838287, 294.1506348, -460.2499390, 447.8580017
3: -285.4850464, 232.3374176, -324.0071411, 264.3378601, -549.8228760, 556.3445435
4: -210.4840851, 260.5068970, -238.8877716, 295.7174072, -506.2014771, 499.3945923

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794710
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794762
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -81.8196487, 68.6777191, -92.6804199, 77.8872147, -159.7068634, 161.3581390
1: -311.6664429, 255.5859375, -353.1406555, 290.3778381, -602.0443115, 608.7265625
2: -167.0954895, 260.6704407, -189.0595398, 294.8270874, -461.9225769, 449.7299805
3: -287.1930237, 233.7344971, -324.8352051, 265.2995911, -552.4926147, 558.5696411
4: -211.7368164, 262.1036682, -239.3786163, 296.3341980, -508.0710144, 501.4822998

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -81.0548477, 68.0324707, -96.5716095, 81.0334015, -162.0882416, 164.6040802
1: -308.7578125, 253.1779633, -367.7864075, 302.4145813, -611.1723022, 620.9643555
2: -165.5392609, 258.2379150, -197.2401733, 306.0216980, -471.5609436, 455.4780884
3: -284.5516357, 231.5315247, -338.4599304, 276.1612549, -560.7128906, 569.9913330
4: -209.7725830, 259.6732483, -249.7925720, 307.5279236, -517.3005371, 509.4657898

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -85.1929016, 71.4209976, -96.5716095, 81.0334015, -166.2262726, 167.9925995
1: -324.3200684, 265.9903870, -367.7864075, 302.4145813, -626.7346191, 633.7767944
2: -174.3341370, 270.4047546, -197.2401733, 306.0216980, -480.3557434, 467.6449280
3: -298.9140320, 243.2106781, -338.4599304, 276.1612549, -575.0753174, 581.6705322
4: -220.7886353, 271.9003296, -249.7925720, 307.5279236, -528.3165283, 521.6928711

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0795035
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0795035
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -92.4449081, 77.6091919, -81.3296356, 68.2649765, -160.7098846, 158.9388275
1: -352.0039978, 289.2525024, -309.7753296, 254.0611267, -606.0651245, 599.0278320
2: -188.7838287, 294.1506348, -166.0993042, 259.0742188, -447.8580017, 460.2499390
3: -324.0071411, 264.3378601, -285.4850464, 232.3374176, -556.3445435, 549.8228760
4: -238.8877716, 295.7174072, -210.4840851, 260.5068970, -499.3945923, 506.2014771

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798682
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798898
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -92.6804199, 77.8872147, -81.8196487, 68.6777191, -161.3581390, 159.7068634
1: -353.1406555, 290.3778381, -311.6664429, 255.5859375, -608.7265015, 602.0443115
2: -189.0595398, 294.8270874, -167.0954895, 260.6704407, -449.7299805, 461.9225769
3: -324.8352051, 265.2995911, -287.1930237, 233.7344971, -558.5696411, 552.4926147
4: -239.3786163, 296.3341980, -211.7368164, 262.1036682, -501.4822998, 508.0710144

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -96.5716095, 81.0334015, -81.0548477, 68.0324707, -164.6040802, 162.0882416
1: -367.7864075, 302.4145813, -308.7578125, 253.1779633, -620.9643555, 611.1723022
2: -197.2401733, 306.0216980, -165.5392609, 258.2379150, -455.4780884, 471.5609436
3: -338.4599304, 276.1612549, -284.5516357, 231.5315247, -569.9913330, 560.7128906
4: -249.7925720, 307.5279236, -209.7725830, 259.6732483, -509.4657898, 517.3005371

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794801, upper bound: 141.0797590
time: 0.91 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -96.5716095, 81.0334015, -85.1929016, 71.4209976, -167.9925995, 166.2262726
1: -367.7864075, 302.4145813, -324.3200684, 265.9903870, -633.7767944, 626.7346191
2: -197.2401733, 306.0216980, -174.3341370, 270.4047546, -467.6449280, 480.3557434
3: -338.4599304, 276.1612549, -298.9140320, 243.2106781, -581.6705322, 575.0753174
4: -249.7925720, 307.5279236, -220.7886353, 271.9003296, -521.6928711, 528.3165283

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -91.2297058, 76.5480728, -92.1772385, 77.3481293, -168.5778351, 168.7253113
1: -347.7825317, 285.1564331, -351.3999329, 288.1518250, -635.9342651, 636.5562744
2: -186.0379486, 289.9358215, -187.9627686, 292.9424133, -478.9803467, 477.8985596
3: -319.8093567, 260.6633911, -323.0997009, 263.3986511, -583.2080078, 583.7630615
4: -235.6557159, 291.3658142, -238.0829163, 294.3644714, -530.0201416, 529.4487305

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0792499
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0795397
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -94.7634583, 79.3945465, -92.7058411, 77.7951508, -172.5585785, 172.1003876
1: -361.0857544, 296.1223450, -353.4087830, 289.8372192, -650.9229736, 649.5311279
2: -193.5030212, 300.0057068, -189.0710449, 294.5179138, -488.0209351, 489.0767517
3: -332.1995239, 270.5268555, -324.9212341, 264.9313049, -597.1307373, 595.4480591
4: -245.1872253, 301.4217529, -239.4346161, 295.8177795, -541.0050049, 540.8562622

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794179, upper bound: 141.0792737
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795624, upper bound: 141.0795624
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -91.2297058, 76.5480728, -121.5814209, 102.0380783, -193.2677917, 198.1294861
1: -347.7825317, 285.1564331, -464.9477844, 379.9028015, -727.6852417, 750.1042480
2: -186.0379486, 289.9358215, -245.6571808, 385.5071411, -571.5451050, 535.5928955
3: -319.8093567, 260.6633911, -426.2301025, 346.9422913, -666.7516479, 686.8934937
4: -235.6557159, 291.3658142, -313.5905762, 386.1867371, -621.8424072, 604.9564209

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791105, upper bound: 141.0789761
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794472, upper bound: 141.0794279
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794472, upper bound: 141.0794508
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -94.7634583, 79.3945465, -121.9798737, 102.3745117, -197.1379700, 201.3744202
1: -361.0857544, 296.1223450, -466.4725342, 381.1499023, -742.2356567, 762.5948486
2: -193.5030212, 300.0057068, -246.4880981, 386.8665466, -580.3693848, 546.4937744
3: -332.1995239, 270.5268555, -427.6000061, 348.0472412, -680.2467041, 698.1267700
4: -245.1872253, 301.4217529, -314.6008606, 387.5271606, -632.7143555, 616.0224609

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0793949
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794523, upper bound: 141.0794761
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -121.5814209, 102.0380783, -91.2297058, 76.5480728, -198.1294861, 193.2677917
1: -464.9477844, 379.9028015, -347.7825317, 285.1564331, -750.1042480, 727.6852417
2: -245.6571808, 385.5071411, -186.0379486, 289.9358215, -535.5928955, 571.5451050
3: -426.2301025, 346.9422913, -319.8093567, 260.6633911, -686.8934937, 666.7516479
4: -313.5905762, 386.1867371, -235.6557159, 291.3658142, -604.9564209, 621.8424072

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793600, upper bound: 141.0793551
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794441, upper bound: 141.0795428
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794613, upper bound: 141.0795478
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -121.9798737, 102.3745117, -94.7634583, 79.3945465, -201.3744202, 197.1379700
1: -466.4725342, 381.1499023, -361.0857544, 296.1223450, -762.5948486, 742.2356567
2: -246.4880981, 386.8665466, -193.5030212, 300.0057068, -546.4937134, 580.3695068
3: -427.6000061, 348.0472412, -332.1995239, 270.5268555, -698.1267700, 680.2467041
4: -314.6008606, 387.5271606, -245.1872253, 301.4217529, -616.0224609, 632.7142944

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794376, upper bound: 141.0795097
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794613, upper bound: 141.0795464
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -120.3879929, 101.2533417, -115.9942017, 97.8066635, -218.1946259, 217.2475433
1: -459.9667358, 377.1454468, -442.5007629, 364.3755798, -824.3421021, 819.6460571
2: -243.6004028, 382.9964600, -234.9500275, 369.5809021, -613.1812744, 617.9464722
3: -421.6700134, 344.2608643, -405.9994812, 332.5202332, -754.1902466, 750.2603149
4: -310.3226013, 383.6067200, -298.8660889, 370.1399231, -680.4624634, 682.4727783

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779931
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -121.5270615, 102.0146103, -120.4044724, 101.0756378, -222.6026611, 222.4190826
1: -464.6864319, 379.7958679, -460.3747253, 376.2780457, -840.9644775, 840.1704712
2: -245.6232300, 385.5372314, -243.4151917, 382.0302429, -627.6533813, 628.9523926
3: -426.0509644, 346.8414307, -422.1693115, 343.6130981, -769.6640625, 769.0107422
4: -313.4792175, 386.2083435, -310.6288452, 382.6932678, -696.1724854, 696.8371582

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779934, upper bound: 141.0792470
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779934, upper bound: 141.0794888
time: 0.70 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.75 seconds
NS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794710
NS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794762
NS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
NS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0795035
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0795035
NS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798682
NS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798898
NS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
NS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
NS_A2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794801, upper bound: 141.0797590
NS_A2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
NS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
NS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794869, upper bound: 141.0798248
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0792499
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0795397
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794179, upper bound: 141.0792737
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0795624, upper bound: 141.0795624
NS_A2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794472, upper bound: 141.0794279
NS_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794472, upper bound: 141.0794508
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0793949
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794523, upper bound: 141.0794761
NS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794441, upper bound: 141.0795428
NS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794613, upper bound: 141.0795478
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794376, upper bound: 141.0795097
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0794613, upper bound: 141.0795464
NS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
NS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779931
NS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0779934, upper bound: 141.0792470
NS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 0, lower bound: -141.0779934, upper bound: 141.0794888

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -80.4164047, 67.4943924, -92.4449081, 77.6091919, -158.0256042, 159.9393005
1: -306.2994995, 251.1833649, -352.0039978, 289.2525024, -595.5520020, 603.1873779
2: -164.2514648, 256.1733704, -188.7838287, 294.1506348, -458.4020996, 444.9571838
3: -282.3259277, 229.7065735, -324.0071411, 264.3378601, -546.6638184, 553.7137451
4: -208.1423492, 257.6064758, -238.8877716, 295.7174072, -503.8597412, 496.4942627

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794710
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794710
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -84.5428009, 70.8735428, -92.4449081, 77.6091919, -162.1519928, 163.3184204
1: -321.8095093, 263.9619141, -352.0039978, 289.2525024, -611.0618896, 615.9659424
2: -173.0247498, 268.3034668, -188.7838287, 294.1506348, -467.1753540, 457.0872803
3: -296.6412964, 241.3556824, -324.0071411, 264.3378601, -560.9791260, 565.3627930
4: -219.1285706, 269.7967224, -238.8877716, 295.7174072, -514.8459473, 508.6844788

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794762
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798682, upper bound: 141.0794762
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -81.5346832, 68.4639053, -92.6804199, 77.8872147, -159.4219055, 161.1443176
1: -310.2488708, 254.6724091, -353.1406555, 290.3778381, -600.6266479, 607.8130493
2: -166.8037415, 259.8311462, -189.0595398, 294.8270874, -461.6308289, 448.8906860
3: -286.1627197, 232.8170776, -324.8352051, 265.2995911, -551.4622803, 557.6522827
4: -211.1346893, 261.3152466, -239.3786163, 296.3341980, -507.4688721, 500.6938477

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -81.4205856, 68.3401337, -92.6804199, 77.8872147, -159.3078003, 161.0205536
1: -310.1306458, 254.3217163, -353.1406555, 290.3778381, -600.5084839, 607.4624023
2: -166.3042297, 259.4004822, -189.0595398, 294.8270874, -461.1313171, 448.4599915
3: -285.8014221, 232.5818176, -324.8352051, 265.2995911, -551.1009521, 557.4169922
4: -210.7222290, 260.8301086, -239.3786163, 296.3341980, -507.0564270, 500.2087402

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792642, upper bound: 141.0788431
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -81.0548477, 68.0324707, -94.7634583, 79.3945465, -160.4494019, 162.7959290
1: -308.7578125, 253.1779633, -361.0857544, 296.1223450, -604.8801270, 614.2635498
2: -165.5392609, 258.2379150, -193.5030212, 300.0057068, -465.5449524, 451.7409363
3: -284.5516357, 231.5315247, -332.1995239, 270.5268555, -555.0784912, 563.7309570
4: -209.7725830, 259.6732483, -245.1872253, 301.4217529, -511.1943359, 504.8604431

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797590, upper bound: 141.0794801
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798248, upper bound: 141.0794869
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.0548477, 68.0324707, -122.0526886, 102.2769547, -183.3318024, 190.0851593
1: -308.7578125, 253.1779633, -466.5812378, 380.7080078, -689.4656372, 719.7591553
2: -165.5392609, 258.2379150, -247.0280304, 386.1335449, -551.6727905, 505.2659302
3: -284.5516357, 231.5315247, -428.0799866, 347.5617676, -632.1133423, 659.6113892
4: -209.7725830, 259.6732483, -315.3374939, 386.7519836, -596.5245361, 575.0104980

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797590, upper bound: 141.0794801
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796077, upper bound: 141.0793447
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797070, upper bound: 141.0793741
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -85.1929016, 71.4209976, -94.7634583, 79.3945465, -164.5874481, 166.1844025
1: -324.3200684, 265.9903870, -361.0857544, 296.1223450, -620.4423828, 627.0761719
2: -174.3341370, 270.4047546, -193.5030212, 300.0057068, -474.3398132, 463.9077759
3: -298.9140320, 243.2106781, -332.1995239, 270.5268555, -569.4409180, 575.4100952
4: -220.7886353, 271.9003296, -245.1872253, 301.4217529, -522.2103271, 517.0875244

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798504, upper bound: 141.0794921
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799309, upper bound: 141.0795035
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -85.1929016, 71.4209976, -122.0526886, 102.2769547, -187.4698334, 193.4736786
1: -324.3200684, 265.9903870, -466.5812378, 380.7080078, -705.0279541, 732.5716553
2: -174.3341370, 270.4047546, -247.0280304, 386.1335449, -560.4676514, 517.4328003
3: -298.9140320, 243.2106781, -428.0799866, 347.5617676, -646.4758301, 671.2905884
4: -220.7886353, 271.9003296, -315.3374939, 386.7519836, -607.5405273, 587.2376709

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798504, upper bound: 141.0794921
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799309, upper bound: 141.0795035
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -92.4449081, 77.6091919, -80.4164047, 67.4943924, -159.9393005, 158.0256042
1: -352.0039978, 289.2525024, -306.2994995, 251.1833649, -603.1873779, 595.5520020
2: -188.7838287, 294.1506348, -164.2514648, 256.1733704, -444.9571838, 458.4020996
3: -324.0071411, 264.3378601, -282.3259277, 229.7065735, -553.7137451, 546.6638184
4: -238.8877716, 295.7174072, -208.1423492, 257.6064758, -496.4942627, 503.8597412

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798682
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798682
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -92.4449081, 77.6091919, -84.5428009, 70.8735428, -163.3184204, 162.1519928
1: -352.0039978, 289.2525024, -321.8095093, 263.9619141, -615.9659424, 611.0618896
2: -188.7838287, 294.1506348, -173.0247498, 268.3034668, -457.0872803, 467.1753540
3: -324.0071411, 264.3378601, -296.6412964, 241.3556824, -565.3627930, 560.9791260
4: -238.8877716, 295.7174072, -219.1285706, 269.7967224, -508.6844788, 514.8459473

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798898
time: 1.22 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794710, upper bound: 141.0798898
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -92.6804199, 77.8872147, -81.5346832, 68.4639053, -161.1443176, 159.4218903
1: -353.1406555, 290.3778381, -310.2488708, 254.6724091, -607.8130493, 600.6267090
2: -189.0595398, 294.8270874, -166.8037415, 259.8311462, -448.8906860, 461.6308289
3: -324.8352051, 265.2995911, -286.1627197, 232.8170776, -557.6522827, 551.4622803
4: -239.3786163, 296.3341980, -211.1346893, 261.3152466, -500.6938477, 507.4688721

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -92.6804199, 77.8872147, -81.4205856, 68.3401337, -161.0205536, 159.3078003
1: -353.1406555, 290.3778381, -310.1306458, 254.3217163, -607.4624023, 600.5084839
2: -189.0595398, 294.8270874, -166.3042297, 259.4004822, -448.4599915, 461.1313171
3: -324.8352051, 265.2995911, -285.8014221, 232.5818176, -557.4169922, 551.1009521
4: -239.3786163, 296.3341980, -210.7222290, 260.8301086, -500.2087402, 507.0564270

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788431, upper bound: 141.0792642
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -96.5716095, 81.0334015, -80.6269226, 67.6673965, -164.2390137, 161.6603088
1: -367.7864075, 302.4145813, -307.1433105, 251.8518829, -619.6383057, 609.5578613
2: -197.2401733, 306.0216980, -164.6431427, 256.8087769, -454.0489502, 470.6647949
3: -338.4599304, 276.1612549, -283.0576172, 230.3236847, -568.7835083, 559.2188721
4: -249.7925720, 307.5279236, -208.6735535, 258.2519226, -508.0444641, 516.2014771

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794400, upper bound: 141.0797518
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794400, upper bound: 141.0797590
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -96.5716095, 81.0334015, -80.4419861, 67.4939575, -164.0655670, 161.4753723
1: -367.7864075, 302.4145813, -306.5677185, 251.2015839, -618.9879761, 608.9822998
2: -197.2401733, 306.0216980, -164.2011108, 256.0919189, -453.3320923, 470.2227478
3: -338.4599304, 276.1612549, -282.4529419, 229.7292328, -568.1889648, 558.6141968
4: -249.7925720, 307.5279236, -208.1888275, 257.5232544, -507.3158264, 515.7167358

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794469, upper bound: 141.0798174
time: 0.70 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794469, upper bound: 141.0798248
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -94.7634583, 79.3945465, -85.1929016, 71.4209976, -166.1844177, 164.5874481
1: -361.0857544, 296.1223450, -324.3200684, 265.9903870, -627.0761108, 620.4423828
2: -193.5030212, 300.0057068, -174.3341370, 270.4047546, -463.9077759, 474.3398132
3: -332.1995239, 270.5268555, -298.9140320, 243.2106781, -575.4100952, 569.4408569
4: -245.1872253, 301.4217529, -220.7886353, 271.9003296, -517.0875244, 522.2103271

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794802, upper bound: 141.0797590
time: 0.75 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0798248
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.0526886, 102.2769547, -85.1929016, 71.4209976, -193.4736786, 187.4698334
1: -466.5812378, 380.7080078, -324.3200684, 265.9903870, -732.5716553, 705.0279541
2: -247.0280304, 386.1335449, -174.3341370, 270.4047546, -517.4328003, 560.4676514
3: -428.0799866, 347.5617676, -298.9140320, 243.2106781, -671.2905884, 646.4758301
4: -315.3374939, 386.7519836, -220.7886353, 271.9003296, -587.2376709, 607.5405273

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794802, upper bound: 141.0797590
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0798248
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -88.0829620, 73.8946304, -85.8902054, 72.0030441, -160.0859833, 159.7847900
1: -335.7076416, 275.2825317, -327.4373169, 267.9023438, -603.6099243, 602.7198486
2: -179.6896362, 279.9587708, -175.1765747, 273.4515381, -453.1411133, 455.1353149
3: -308.8486328, 251.6277313, -300.9812317, 245.0772247, -553.9258423, 552.6089478
4: -227.6392975, 281.4376526, -221.7282104, 274.7578735, -502.3971252, 503.1658630

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0792499
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794172, upper bound: 141.0792499
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -90.8379822, 76.2199631, -91.5089645, 76.7884598, -167.6264343, 167.7289276
1: -346.2824707, 283.9381104, -348.8399963, 286.0731506, -632.3555908, 632.7780762
2: -185.2555237, 288.6996155, -186.6285248, 290.8336792, -476.0891724, 475.3281250
3: -318.4376526, 259.5506592, -320.7589417, 261.5009766, -579.9385986, 580.3095703
4: -234.6573944, 290.1366272, -236.3795776, 292.2679138, -526.9252930, 526.5160522

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795596, upper bound: 141.0795226
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795596, upper bound: 141.0795397
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -91.6191940, 76.7568741, -86.3668976, 72.4138794, -164.0330505, 163.1237793
1: -349.0016479, 286.2780762, -329.2604675, 269.4892883, -618.4908447, 615.5385742
2: -187.2112732, 290.1046448, -176.1871185, 275.1199341, -462.3312073, 466.2917480
3: -321.2090149, 261.5663452, -302.6230469, 246.4952393, -567.7042236, 564.1893921
4: -237.1560974, 291.6008606, -222.9428558, 276.4138794, -513.5699463, 514.5437012

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794179, upper bound: 141.0792737
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794179, upper bound: 141.0792737
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -94.3851166, 79.0762558, -92.0320206, 77.2314148, -171.6165161, 171.1082764
1: -359.6382446, 294.9393311, -350.8260498, 287.7453918, -647.3835449, 645.7653198
2: -192.7504272, 298.8098755, -187.7273102, 292.4105530, -485.1608582, 486.5371704
3: -330.8762817, 269.4445496, -322.5594177, 263.0199585, -593.8962402, 592.0039062
4: -244.2245178, 300.2291260, -237.7177429, 293.6993713, -537.9238892, 537.9468994

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0782108, upper bound: 141.0793257
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795580, upper bound: 141.0795580
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -84.9730835, 71.2224045, -118.4599915, 99.3881760, -184.3612671, 189.6823883
1: -323.9505920, 264.9670410, -452.9241333, 370.0634460, -694.0140381, 717.8911133
2: -173.3084259, 270.4489136, -239.4164734, 375.2743835, -548.5828247, 509.8652954
3: -297.7947083, 242.4077148, -415.3758850, 337.9708252, -635.7655029, 657.7835693
4: -219.3752747, 271.7419434, -305.6520081, 376.0334167, -595.4085693, 577.3939209

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793963, upper bound: 141.0794116
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794472, upper bound: 141.0794279
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -90.5662155, 75.9926987, -121.1685410, 101.6798782, -192.2460938, 197.1612091
1: -345.2415161, 283.0943298, -463.3803101, 378.5232544, -723.7647705, 746.4744873
2: -184.7126923, 287.8437805, -244.8132019, 384.1709900, -568.8836670, 532.6568604
3: -317.4857178, 258.7800293, -424.8021240, 345.6550598, -663.1407471, 683.5821533
4: -233.9644623, 289.2855225, -312.5403748, 384.8336792, -618.7981567, 601.8259277

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793979, upper bound: 141.0794338
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794508, upper bound: 141.0794508
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -91.6191940, 76.7568741, -117.9843597, 99.0273361, -190.6465149, 194.7412109
1: -349.0016479, 286.2780762, -451.1291504, 368.1656799, -717.1673584, 737.4072266
2: -187.2112732, 290.1046448, -238.4149323, 374.2479248, -561.4591675, 528.5195923
3: -321.2090149, 261.5663452, -413.5065002, 336.3186951, -657.5277100, 675.0727539
4: -237.1560974, 291.6008606, -304.3164368, 375.0194092, -612.1755371, 595.9172974

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793381, upper bound: 141.0793537
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793855, upper bound: 141.0793816
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -94.3851166, 79.0762558, -121.3113556, 101.7952652, -196.1803894, 200.3876038
1: -359.6382446, 294.9393311, -463.9300842, 378.9293823, -738.5676270, 758.8693237
2: -192.7504272, 298.8098755, -245.1396332, 384.6985779, -577.4489746, 543.9495239
3: -330.8762817, 269.4445496, -425.2815552, 345.9786072, -676.8548584, 694.7260742
4: -244.2245178, 300.2291260, -312.8969116, 385.3389282, -629.5634766, 613.1260376

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794018, upper bound: 141.0794676
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794523, upper bound: 141.0794761
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -118.4599915, 99.3881760, -84.9730835, 71.2224045, -189.6824036, 184.3612671
1: -452.9241333, 370.0634460, -323.9505920, 264.9670410, -717.8911743, 694.0140381
2: -239.4164734, 375.2743835, -173.3084259, 270.4489136, -509.8652954, 548.5828247
3: -415.3758850, 337.9708252, -297.7947083, 242.4077148, -657.7835083, 635.7655029
4: -305.6520081, 376.0334167, -219.3752747, 271.7419434, -577.3939209, 595.4085693

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794397, upper bound: 141.0794971
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794441, upper bound: 141.0795428
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -121.1685410, 101.6798782, -90.5662155, 75.9926987, -197.1611938, 192.2460938
1: -463.3803101, 378.5232544, -345.2415161, 283.0943298, -746.4746094, 723.7647705
2: -244.8132019, 384.1709900, -184.7126923, 287.8437805, -532.6568604, 568.8836060
3: -424.8021240, 345.6550598, -317.4857178, 258.7800293, -683.5821533, 663.1407471
4: -312.5403748, 384.8336792, -233.9644623, 289.2855225, -601.8259277, 618.7981567

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794558, upper bound: 141.0795000
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794613, upper bound: 141.0795478
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -117.9843597, 99.0273361, -91.6191940, 76.7568741, -194.7411957, 190.6464996
1: -451.1291504, 368.1656799, -349.0016479, 286.2780762, -737.4072266, 717.1673584
2: -238.4149323, 374.2479248, -187.2112732, 290.1046448, -528.5195923, 561.4591675
3: -413.5065002, 336.3186951, -321.2090149, 261.5663452, -675.0728149, 657.5277100
4: -304.3164368, 375.0194092, -237.1560974, 291.6008606, -595.9172974, 612.1755371

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794288, upper bound: 141.0794621
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794376, upper bound: 141.0795097
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -121.3113556, 101.7952652, -94.3851166, 79.0762558, -200.3876038, 196.1803894
1: -463.9300842, 378.9293823, -359.6382446, 294.9393311, -758.8693237, 738.5675659
2: -245.1396332, 384.6985779, -192.7504272, 298.8098755, -543.9495239, 577.4489746
3: -425.2815552, 345.9786072, -330.8762817, 269.4445496, -694.7260742, 676.8548584
4: -312.8969116, 385.3389282, -244.2245178, 300.2291260, -613.1260376, 629.5634766

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794074, upper bound: 141.0792712
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794074, upper bound: 141.0795464
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -115.9942017, 97.8066635, -115.9942017, 97.8066635, -213.8008575, 213.8008575
1: -442.5007629, 364.3755798, -442.5007629, 364.3755798, -806.8760986, 806.8760376
2: -234.9500275, 369.5809021, -234.9500275, 369.5809021, -604.5309448, 604.5309448
3: -405.9994812, 332.5202332, -405.9994812, 332.5202332, -738.5197144, 738.5197144
4: -298.8660889, 370.1399231, -298.8660889, 370.1399231, -669.0057983, 669.0057983

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779697, upper bound: 141.0778322
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -120.4044724, 101.0756378, -115.9942017, 97.8066635, -218.2111359, 217.0698242
1: -460.3747253, 376.2780457, -442.5007629, 364.3755798, -824.7501221, 818.7787476
2: -243.4151917, 382.0302429, -234.9500275, 369.5809021, -612.9960938, 616.9802856
3: -422.1693115, 343.6130981, -405.9994812, 332.5202332, -754.6895752, 749.6125488
4: -310.6288452, 382.6932678, -298.8660889, 370.1399231, -680.7686768, 681.5592041

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778117, upper bound: 141.0779694
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779931
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -115.9942017, 97.8066635, -120.4044724, 101.0756378, -217.0698242, 218.2111359
1: -442.5007629, 364.3755798, -460.3747253, 376.2780457, -818.7788086, 824.7501221
2: -234.9500275, 369.5809021, -243.4151917, 382.0302429, -616.9802856, 612.9960938
3: -405.9994812, 332.5202332, -422.1693115, 343.6130981, -749.6125488, 754.6895752
4: -298.8660889, 370.1399231, -310.6288452, 382.6932678, -681.5592651, 680.7686768

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779697, upper bound: 141.0791714
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0792470
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -120.4044724, 101.0756378, -120.4044724, 101.0756378, -221.4801025, 221.4801025
1: -460.3747253, 376.2780457, -460.3747253, 376.2780457, -836.6527710, 836.6527710
2: -243.4151917, 382.0302429, -243.4151917, 382.0302429, -625.4454346, 625.4454346
3: -422.1693115, 343.6130981, -422.1693115, 343.6130981, -765.7824097, 765.7824097
4: -310.6288452, 382.6932678, -310.6288452, 382.6932678, -693.3220825, 693.3220825

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778118, upper bound: 141.0794693
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.99 + 221.35 = 224.34 seconds
