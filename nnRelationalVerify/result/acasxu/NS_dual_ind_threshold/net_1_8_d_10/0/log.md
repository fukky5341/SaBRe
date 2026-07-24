## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 141.077538292802


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
execution time: IAR + RelationalAnalysis = 1.03 + 2.10 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -141.0803599, upper bound: 141.0803599

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800760, upper bound: 141.0797630
time: 0.78 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797537, upper bound: 141.0797537
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -141.0800760, upper bound: 141.0797630
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -141.0797537, upper bound: 141.0797537

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -86.4030304, 72.6540375, -88.3026199, 74.4034729, -160.8064728, 160.9566650
1: -328.9674072, 270.5637207, -336.0358887, 277.3612366, -606.3286133, 606.5995483
2: -176.6538391, 275.5844116, -180.5916595, 281.9598389, -458.6136780, 456.1760864
3: -303.0539246, 247.2805328, -309.5568848, 253.4273224, -556.4812622, 556.8374023
4: -223.4498444, 276.9497986, -228.2478943, 283.7137756, -507.1635437, 505.1976929

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0743358, upper bound: 141.0635068
time: 0.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795601, upper bound: 141.0794972
time: 0.80 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -112.5245056, 94.6374817, -88.1435242, 74.3931656, -186.9176636, 182.7810059
1: -429.9682312, 352.5945129, -334.9487000, 277.8852234, -707.8533936, 687.5432129
2: -228.2848358, 356.8963013, -180.0106049, 281.6024170, -509.8872681, 536.9069214
3: -394.6705322, 321.9489746, -308.5078430, 253.5371552, -648.2077026, 630.4567871
4: -290.5984802, 357.8221741, -227.4971619, 283.5516357, -574.1501465, 585.3193359

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0740116, upper bound: 141.0634861
time: 0.75 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794894, upper bound: 141.0794894
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.52 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -141.0743358, upper bound: 141.0635068
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -141.0795601, upper bound: 141.0794972
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -141.0740116, upper bound: 141.0634861
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -141.0794894, upper bound: 141.0794894

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -81.0857468, 68.1835022, -93.3808975, 78.4782944, -159.5640411, 161.5643921
1: -308.7213745, 254.2777557, -355.7897949, 292.5882568, -601.3095093, 610.0675659
2: -165.5245819, 258.1622009, -190.4549713, 297.0440674, -462.5686646, 448.6171875
3: -284.2078552, 232.3271484, -327.3085938, 267.3828125, -551.5906982, 559.6357422
4: -209.5800629, 259.8278198, -241.1539764, 298.6288757, -508.2089233, 500.9818115

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795435, upper bound: 141.0794573
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795548, upper bound: 141.0794953
time: 0.80 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -107.8232346, 90.7144852, -93.6438980, 78.8366623, -186.6598511, 184.3583832
1: -411.9169312, 338.1351318, -356.6582031, 294.2830811, -706.1998901, 694.7933350
2: -218.5886841, 342.1151733, -190.7530365, 297.8670654, -516.4556885, 532.8681641
3: -378.3233948, 308.6474915, -328.2691650, 268.5314636, -646.8548584, 636.9165649
4: -278.5840454, 343.1355591, -241.8630066, 299.5710144, -578.1549683, 584.9985352

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794861, upper bound: 141.0794573
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0794871
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.79 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -141.0795435, upper bound: 141.0794573
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -141.0795548, upper bound: 141.0794953
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -141.0794861, upper bound: 141.0794573
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0794871

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -73.5098877, 61.8138771, -90.2600021, 75.8509140, -149.3607788, 152.0738831
1: -279.9388428, 230.1178284, -343.8036194, 282.8242188, -562.7630615, 573.9214478
2: -150.1599731, 234.7194824, -184.1538391, 287.2031555, -437.3631287, 418.8732910
3: -257.8670349, 210.5574646, -316.4202881, 258.4459534, -516.3128662, 526.9777832
4: -190.0821228, 236.0159454, -233.1887207, 288.7961121, -478.8782349, 469.2046509

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795435, upper bound: 141.0794573
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795435, upper bound: 141.0794573
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -80.3107071, 67.5416946, -92.9791489, 78.1410370, -158.4517365, 160.5208435
1: -305.7069092, 251.8857574, -354.2560730, 291.3348389, -597.0417480, 606.1417847
2: -164.0178375, 255.7444611, -189.6555786, 295.7756348, -459.7934570, 445.4000244
3: -281.4871826, 230.1256714, -325.9078369, 266.2383423, -547.7255249, 556.0334473
4: -207.6402435, 257.4139404, -240.1365967, 297.3648682, -505.0051270, 497.5504150

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792470, upper bound: 141.0779920
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795548, upper bound: 141.0794927
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -102.5557556, 86.2392044, -90.8321457, 76.4919205, -179.0476685, 177.0713196
1: -391.7066650, 321.3295898, -345.6870422, 285.5665283, -677.2730713, 667.0166016
2: -207.8349762, 325.5191345, -185.0785828, 288.9691467, -496.8041077, 510.5977173
3: -359.7174072, 293.5523682, -318.2549744, 260.5735779, -620.2910156, 611.8071289
4: -264.8685913, 326.3827820, -234.5602722, 290.8534546, -555.7219849, 560.9430542

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794861, upper bound: 141.0794573
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794861, upper bound: 141.0794573
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -107.0948181, 90.0664597, -93.1270981, 78.4035645, -185.4983826, 183.1935577
1: -409.1457825, 335.6411438, -354.6896973, 292.6591492, -701.8049316, 690.3308105
2: -217.1208496, 339.6615601, -189.7263489, 296.2542419, -513.3750610, 529.3878784
3: -375.8165588, 306.3641052, -326.4898376, 267.0444336, -642.8609619, 632.8538208
4: -276.7417297, 340.5817261, -240.5632629, 297.9415588, -574.6831665, 581.1449585

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0794871
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0794871
time: 0.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.66 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0795435, upper bound: 141.0794573
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0795435, upper bound: 141.0794573
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0792470, upper bound: 141.0779920
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0795548, upper bound: 141.0794927
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0794861, upper bound: 141.0794573
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0794861, upper bound: 141.0794573
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0794871
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 0, lower bound: -141.0794871, upper bound: 141.0794871

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -73.5098877, 61.8138771, -88.2928085, 74.0725479, -147.5824280, 150.1066742
1: -279.9388428, 230.1178284, -336.4912720, 275.9961243, -555.9349365, 566.6090698
2: -150.1599731, 234.7194824, -180.1186218, 280.6455383, -430.8055115, 414.8381042
3: -257.8670349, 210.5574646, -309.5953064, 252.3220825, -510.1891174, 520.1527710
4: -190.0821228, 236.0159454, -228.1729431, 282.1822205, -472.2643433, 464.1889038

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795428, upper bound: 141.0794441
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792712, upper bound: 141.0794074
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -73.5098877, 61.8138771, -119.9090958, 100.6292114, -174.1390991, 181.7229767
1: -279.9388428, 230.1178284, -458.4467468, 374.7141418, -654.6528931, 688.5645752
2: -150.1599731, 234.7194824, -242.3562622, 379.9991150, -530.1590576, 477.0757446
3: -257.8670349, 210.5574646, -420.3896484, 342.2195740, -600.0866089, 630.9471436
4: -190.0821228, 236.0159454, -309.3559570, 380.7731018, -570.8551636, 545.3718872

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795428, upper bound: 141.0794441
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792712, upper bound: 141.0794074
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -76.8202057, 64.7904892, -85.3082886, 72.2420197, -149.0622253, 150.0987549
1: -291.8193665, 241.6270752, -323.8516235, 269.2095947, -561.0289307, 565.4786987
2: -157.3004913, 245.7747345, -174.7890930, 274.4204712, -431.7209473, 420.5638428
3: -268.8691101, 220.6766510, -298.3237915, 245.9025879, -514.7716675, 519.0004272
4: -198.3912659, 247.2054443, -219.9095306, 275.7599487, -474.1512146, 467.1149292

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779920
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -79.5730972, 66.9236374, -91.1854630, 76.6297836, -156.2028809, 158.1091003
1: -302.8327026, 249.5486755, -347.3033447, 285.7816772, -588.6143799, 596.8520508
2: -162.5642548, 253.4323730, -186.0421295, 290.0740051, -452.6382141, 439.4744873
3: -278.9346008, 227.9682770, -319.7072754, 261.0461731, -539.9807739, 547.6754761
4: -205.7204590, 255.0844574, -235.6048279, 291.6038208, -497.3242798, 490.6892700

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0782064, upper bound: 141.0792579
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0782064, upper bound: 141.0794927
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -102.5557556, 86.2392044, -90.4761429, 75.9241409, -178.4798889, 176.7153320
1: -391.7066650, 321.3295898, -344.8149414, 282.8738403, -674.5803223, 666.1445312
2: -207.8349762, 325.5191345, -184.5761108, 287.6036987, -495.4386597, 510.0952148
3: -359.7174072, 293.5523682, -317.1307068, 258.5763855, -618.2938232, 610.6829224
4: -264.8685913, 326.3827820, -233.7574463, 289.0760803, -553.9447021, 560.1401978

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791588, upper bound: 141.0793115
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793644, upper bound: 141.0793452
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -102.5557556, 86.2392044, -119.9090958, 100.6292114, -203.1849670, 206.1483002
1: -391.7066650, 321.3295898, -458.4467468, 374.7141418, -766.4206543, 779.7763672
2: -207.8349762, 325.5191345, -242.3562622, 379.9991150, -587.8341064, 567.8752441
3: -359.7174072, 293.5523682, -420.3896484, 342.2195740, -701.9369507, 713.9420166
4: -264.8685913, 326.3827820, -309.3559570, 380.7731018, -645.6416626, 635.7387695

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791588, upper bound: 141.0793115
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794737, upper bound: 141.0794441
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794297, upper bound: 141.0794312
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -107.0948181, 90.0664597, -93.2219086, 78.2378922, -185.3327026, 183.2883453
1: -409.1457825, 335.6411438, -355.3825989, 291.4812622, -700.6270752, 691.0235596
2: -217.1208496, 339.6615601, -190.0980988, 296.2699890, -513.3907471, 529.7596436
3: -375.8165588, 306.3641052, -326.7207642, 266.4398499, -642.2564087, 633.0847168
4: -276.7417297, 340.5817261, -240.7650604, 297.6807251, -574.4223022, 581.3468018

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779905, upper bound: 141.0792470
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794862, upper bound: 141.0794862
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -107.0948181, 90.0664597, -122.5371628, 102.8484573, -209.9432526, 212.6036224
1: -409.1457825, 335.6411438, -468.6050415, 382.8991089, -792.0449219, 804.2459106
2: -217.1208496, 339.6615601, -247.6020050, 388.6279297, -605.7487793, 587.2635498
3: -375.8165588, 306.3641052, -429.5478210, 349.6511230, -725.4676514, 735.9118652
4: -276.7417297, 340.5817261, -316.0467224, 389.3073425, -666.0490112, 656.6284180

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779905, upper bound: 141.0792470
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794862, upper bound: 141.0794862
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.64 seconds
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0795428, upper bound: 141.0794441
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0792712, upper bound: 141.0794074
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0795428, upper bound: 141.0794441
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0792712, upper bound: 141.0794074
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779920
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0782064, upper bound: 141.0792579
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0782064, upper bound: 141.0794927
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0791588, upper bound: 141.0793115
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0793644, upper bound: 141.0793452
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0794737, upper bound: 141.0794441
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0794297, upper bound: 141.0794312
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0779905, upper bound: 141.0792470
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0794862, upper bound: 141.0794862
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0779905, upper bound: 141.0792470
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -141.0794862, upper bound: 141.0794862

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -71.1795425, 59.8464890, -86.8442154, 72.8437347, -144.0232849, 146.6907043
1: -271.0690918, 222.6710205, -330.9809570, 271.4034729, -542.4725342, 553.6519775
2: -145.4259338, 227.3759918, -177.1574249, 276.0163574, -421.4422913, 404.5334167
3: -249.8483276, 203.7313080, -304.5742493, 248.1203461, -497.9685974, 508.3055420
4: -184.1013947, 228.5519409, -224.4678650, 277.5574951, -461.6588745, 453.0198059

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794872, upper bound: 141.0792418
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794872, upper bound: 141.0795226
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -75.0577393, 63.0145531, -87.3457489, 73.2709045, -148.3285980, 150.3603058
1: -285.7279358, 234.5226898, -332.8699036, 273.0197754, -558.7476807, 567.3925171
2: -153.6804657, 238.6533203, -178.2344818, 277.5105896, -431.1910400, 416.8878174
3: -263.3676453, 214.5286255, -306.2878113, 249.6063232, -512.9739990, 520.8164062
4: -194.4415283, 239.9362793, -225.7471619, 279.0140381, -473.4555664, 465.6834106

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792053, upper bound: 141.0792053
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792053, upper bound: 141.0794857
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -71.1795425, 59.8464890, -118.4599915, 99.3881760, -170.5677185, 178.3064728
1: -271.0690918, 222.6710205, -452.9241333, 370.0634460, -641.1325684, 675.5951538
2: -145.4259338, 227.3759918, -239.4164734, 375.2743835, -520.7003174, 466.7924805
3: -249.8483276, 203.7313080, -415.3758850, 337.9708252, -587.8191528, 619.1071777
4: -184.1013947, 228.5519409, -305.6520081, 376.0334167, -560.1347046, 534.2039795

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794387, upper bound: 141.0793059
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794358, upper bound: 141.0793073
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -75.0577393, 63.0145531, -118.9462891, 99.8107147, -174.8684387, 181.9608459
1: -285.7279358, 234.5226898, -454.7723083, 371.6459961, -657.3738403, 689.2949219
2: -153.6804657, 238.6533203, -240.3857727, 376.9572144, -530.6376953, 479.0390930
3: -263.3676453, 214.5286255, -417.0223389, 339.3869019, -602.7543335, 631.5509033
4: -194.4415283, 239.9362793, -306.8671875, 377.6851501, -572.1266479, 546.8034058

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791460, upper bound: 141.0792707
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791273, upper bound: 141.0792721
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -72.4293976, 61.4979401, -85.3082886, 72.2420197, -144.6714172, 146.8062286
1: -274.5116272, 229.1019897, -323.8516235, 269.2095947, -543.7211914, 552.9536133
2: -148.8453064, 233.9254303, -174.7890930, 274.4204712, -423.2657166, 408.7145081
3: -253.1319733, 209.2257690, -298.3237915, 245.9025879, -499.0345459, 507.5495605
4: -186.7911530, 235.2358093, -219.9095306, 275.7599487, -462.5510864, 455.1453247

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -79.0269623, 66.4630280, -85.3082886, 72.2420197, -151.2689667, 151.7713165
1: -300.7170105, 247.8014984, -323.8516235, 269.2095947, -569.9265747, 571.6531372
2: -161.4815674, 251.7355042, -174.7890930, 274.4204712, -435.9020081, 426.5245972
3: -277.0319214, 226.3745270, -298.3237915, 245.9025879, -522.9345093, 524.6983032
4: -204.2860718, 253.3749542, -219.9095306, 275.7599487, -480.0459900, 473.2844849

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779920
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779920
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -72.4293976, 61.4979401, -91.1854630, 76.6297836, -149.0591583, 152.6834106
1: -274.5116272, 229.1019897, -347.3033447, 285.7816772, -560.2933350, 576.4053345
2: -148.8453064, 233.9254303, -186.0421295, 290.0740051, -438.9192810, 419.9675293
3: -253.1319733, 209.2257690, -319.7072754, 261.0461731, -514.1781616, 528.9330444
4: -186.7911530, 235.2358093, -235.6048279, 291.6038208, -478.3949280, 470.8406372

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0792579
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0792579
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -79.0269623, 66.4630280, -91.1854630, 76.6297836, -155.6567383, 157.6484985
1: -300.7170105, 247.8014984, -347.3033447, 285.7816772, -586.4986572, 595.1048584
2: -161.4815674, 251.7355042, -186.0421295, 290.0740051, -451.5555725, 437.7776489
3: -277.0319214, 226.3745270, -319.7072754, 261.0461731, -538.0781250, 546.0817871
4: -204.2860718, 253.3749542, -235.6048279, 291.6038208, -495.8898315, 488.9797974

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0794643
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0794643
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -101.6661301, 85.2807693, -88.1830826, 73.9732895, -175.6394043, 173.4638519
1: -388.0832825, 317.1436157, -335.9673462, 275.5628967, -663.6460571, 653.1109619
2: -206.1948853, 321.6758118, -180.0041351, 280.0309143, -486.2257690, 501.6799316
3: -356.8251038, 290.0955811, -309.1087036, 251.9406281, -608.7656250, 599.2042847
4: -262.9872742, 322.8632202, -227.9429321, 281.5524292, -544.5396729, 550.8060913

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791338, upper bound: 141.0793891
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791338, upper bound: 141.0793915
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -102.1519775, 85.8980942, -90.3419495, 75.8133926, -177.9653625, 176.2400513
1: -390.1710815, 320.0437622, -344.2944641, 282.4606323, -672.6317139, 664.3382568
2: -207.0178223, 324.2344360, -184.3099060, 287.1928711, -494.2106934, 508.5443420
3: -358.3016968, 292.3731995, -316.6574707, 258.2009277, -616.5026245, 609.0306396
4: -263.8296509, 325.0742188, -233.4111481, 288.6672974, -552.4969482, 558.4853516

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793416, upper bound: 141.0794227
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793416, upper bound: 141.0794254
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -100.0391083, 84.0942383, -118.4599915, 99.3881760, -199.4272766, 202.5542145
1: -382.0672607, 313.2711792, -452.9241333, 370.0634460, -752.1307373, 766.1952515
2: -202.7186890, 317.4246826, -239.4164734, 375.2743835, -577.9930420, 556.8410645
3: -350.9641418, 286.2120667, -415.3758850, 337.9708252, -688.9349365, 701.5879517
4: -258.4129944, 318.3495178, -305.6520081, 376.0334167, -634.4462891, 624.0015259

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794308
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794312
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -101.8747864, 85.5148849, -118.9462891, 99.8107147, -201.6855011, 204.4611664
1: -388.9739380, 318.5489807, -454.7723083, 371.6459961, -760.6199341, 773.3212280
2: -206.9012756, 322.4299622, -240.3857727, 376.9572144, -583.8585205, 562.8157349
3: -357.5431519, 290.9306946, -417.0223389, 339.3869019, -696.9299927, 707.9530029
4: -263.6125793, 323.2859497, -306.8671875, 377.6851501, -641.2977295, 630.1530762

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793870, upper bound: 141.0793870
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794308
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794312
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -99.9498291, 84.5526505, -89.9275436, 75.6763000, -175.6261292, 174.4801636
1: -380.6382446, 315.2061462, -342.3361511, 281.8439331, -662.4820557, 657.5422974
2: -203.3471832, 319.6785278, -183.6649323, 287.0808105, -490.4279785, 503.3434448
3: -350.0321350, 287.5701904, -314.8326111, 257.5615234, -607.5936279, 602.4027100
4: -257.8861084, 320.7027283, -232.0204620, 288.2109070, -546.0970459, 552.7231445

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0781941
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0792470
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -104.6050720, 87.9910889, -92.2213821, 77.3956833, -182.0007629, 180.2124634
1: -399.4220886, 328.0237427, -351.5161743, 288.3669128, -687.7890015, 679.5398560
2: -212.1596222, 331.9995728, -188.0934143, 293.1145020, -505.2741089, 520.0928345
3: -367.0649414, 299.3190308, -323.2658691, 263.5441895, -630.6090698, 622.5848999
4: -270.3831787, 332.9727783, -238.2376709, 294.4787903, -564.8619385, 571.2104492

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794573, upper bound: 141.0795435
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794573, upper bound: 141.0795548
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -99.9498291, 84.5526505, -119.9641266, 100.8876648, -200.8374786, 204.5167542
1: -380.6382446, 315.2061462, -458.3713074, 375.7343445, -756.3724976, 773.5774536
2: -203.3471832, 319.6785278, -242.7429047, 381.6142578, -584.9613647, 562.4213867
3: -350.0321350, 287.5701904, -420.2002258, 342.9468384, -692.9789429, 707.7703857
4: -257.8861084, 320.7027283, -309.2402649, 382.2123108, -640.0982666, 629.9429932

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0792470
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -104.6050720, 87.9910889, -121.0710144, 101.6210632, -206.2261353, 209.0621033
1: -399.4220886, 328.0237427, -462.9439392, 378.2841187, -777.7061768, 790.9676514
2: -212.1596222, 331.9995728, -244.6950226, 384.0620422, -596.2214966, 576.6945190
3: -367.0649414, 299.3190308, -424.4679260, 345.4348450, -712.4996948, 723.7869873
4: -270.3831787, 332.9727783, -312.3185120, 384.7176514, -655.1008301, 645.2912598

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793842, upper bound: 141.0793378
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793674, upper bound: 141.0793674
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.33 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794872, upper bound: 141.0792418
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794872, upper bound: 141.0795226
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0792053, upper bound: 141.0792053
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0792053, upper bound: 141.0794857
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794387, upper bound: 141.0793059
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794358, upper bound: 141.0793073
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0791460, upper bound: 141.0792707
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0791273, upper bound: 141.0792721
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779920
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779920
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0792579
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0792579
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0794643
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0794643
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0791338, upper bound: 141.0793891
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0791338, upper bound: 141.0793915
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0793416, upper bound: 141.0794227
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0793416, upper bound: 141.0794254
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794308
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794312
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794308
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794312
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0781941
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0792470
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794573, upper bound: 141.0795435
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0794573, upper bound: 141.0795548
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0792470
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0793842, upper bound: 141.0793378
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.33
Output dim: 0, lower bound: -141.0793674, upper bound: 141.0793674

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -71.1795425, 59.8464890, -83.6238251, 70.0942383, -141.2737732, 143.4703064
1: -271.0690918, 222.6710205, -318.7680054, 260.7322693, -531.8013306, 541.4390259
2: -145.4259338, 227.3759918, -170.5402527, 266.1124268, -411.5383606, 397.9162598
3: -249.8483276, 203.7313080, -293.1226807, 238.6320343, -488.4803467, 496.8540039
4: -184.1013947, 228.5519409, -215.9122925, 267.5008850, -451.6022949, 444.4642334

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0791541
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794806, upper bound: 141.0792373
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -71.1795425, 59.8464890, -89.3067245, 74.9216156, -146.1011658, 149.1532135
1: -271.0690918, 222.6710205, -340.4296875, 279.1407166, -550.2097168, 563.1007080
2: -145.4259338, 227.3759918, -182.1418915, 283.8090210, -429.2349548, 409.5178833
3: -249.8483276, 203.7313080, -313.1545410, 255.2001953, -505.0484314, 516.8858643
4: -184.1013947, 228.5519409, -230.7479401, 285.3307190, -469.4321289, 459.2998657

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0794330
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794806, upper bound: 141.0795182
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -75.0577393, 63.0145531, -84.0818405, 70.4945297, -145.5522308, 147.0963898
1: -285.7279358, 234.5226898, -320.5139465, 262.2629700, -547.9909058, 555.0366211
2: -153.6804657, 238.6533203, -171.5244293, 267.7322083, -421.4126282, 410.1777344
3: -263.3676453, 214.5286255, -294.6996460, 240.0163879, -503.3840332, 509.2282715
4: -194.4415283, 239.9362793, -217.0838013, 269.1141357, -463.5556335, 457.0200500

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -75.0577393, 63.0145531, -89.8260269, 75.3603363, -150.4180450, 152.8405762
1: -285.7279358, 234.5226898, -342.3911133, 280.7999573, -566.5277710, 576.9136353
2: -153.6804657, 238.6533203, -183.2415771, 285.3023682, -438.9828491, 421.8948975
3: -263.3676453, 214.5286255, -314.9396667, 256.7130432, -520.0806885, 529.4682617
4: -194.4415283, 239.9362793, -232.0801697, 286.7777710, -481.2192993, 472.0163269

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -68.9130402, 57.9228058, -118.2168579, 99.1221466, -168.0351868, 176.1396637
1: -262.2912903, 215.5210114, -451.6958313, 368.5388794, -630.8300781, 667.2168579
2: -140.8688507, 219.9066010, -239.1321411, 373.4939880, -514.3627930, 459.0386658
3: -241.8941650, 197.2267609, -414.7155762, 336.8482971, -578.7424316, 611.9423218
4: -178.3214264, 221.1728821, -305.4321289, 374.4116516, -552.7329712, 526.6049805

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793966, upper bound: 141.0793016
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794387, upper bound: 141.0793059
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -70.9711761, 59.6777458, -118.0864410, 99.0742340, -170.0453796, 177.7641602
1: -270.2513123, 222.0423737, -451.5088196, 368.8962402, -639.1475830, 673.5512085
2: -145.0249939, 226.7467957, -238.6621399, 374.0875854, -519.1124878, 465.4089355
3: -249.1144714, 203.1505585, -414.0669250, 336.8878479, -586.0023193, 617.2174683
4: -183.5706177, 227.9221344, -304.6972046, 374.8463440, -558.4169922, 532.6193237

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790642, upper bound: 141.0792186
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794324, upper bound: 141.0793029
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -72.7638702, 61.0651932, -118.7023849, 99.5487900, -172.3126526, 179.7675781
1: -276.8723450, 227.2120514, -453.5169983, 370.1821899, -647.0544434, 680.7290039
2: -149.1130524, 231.0333405, -240.1113892, 375.1788025, -524.2918701, 471.1446838
3: -255.3299408, 207.9022522, -416.3472290, 338.3166809, -593.6466064, 624.2495117
4: -188.6263580, 232.4588318, -306.6443787, 376.0733948, -564.6997681, 539.1031494

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791460, upper bound: 141.0792707
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -74.9023209, 62.8862534, -118.5102463, 99.4452362, -174.3475647, 181.3964691
1: -285.1292419, 234.0383606, -453.1156006, 370.2891541, -655.4183960, 687.1539307
2: -153.3670349, 238.1658783, -239.5105743, 375.5569458, -528.9239502, 477.6764526
3: -262.8238831, 214.0868530, -415.4940796, 338.1318359, -600.9556274, 629.5809326
4: -194.0394592, 239.4493713, -305.7553406, 376.2937622, -570.3332520, 545.2046509

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791273, upper bound: 141.0792721
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -72.4293976, 61.4979401, -83.3206024, 70.4338150, -142.8632202, 144.8185425
1: -274.5116272, 229.1019897, -316.4634399, 262.3117981, -536.8233643, 545.5654297
2: -148.8453064, 233.9254303, -170.6630859, 267.7605896, -416.6058655, 404.5884705
3: -253.1319733, 209.2257690, -291.4399414, 239.7019501, -492.8339233, 500.6657104
4: -186.7911530, 235.2358093, -214.8449249, 269.0508423, -455.8419495, 450.0807190

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780225, upper bound: 141.0779769
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -72.4293976, 61.4979401, -115.5957718, 97.4592056, -169.8886108, 177.0937195
1: -274.5116272, 229.1019897, -440.9888916, 363.0383606, -637.5499878, 670.0908203
2: -148.8453064, 233.9254303, -234.1324768, 368.2810364, -517.1262817, 468.0578918
3: -253.1319733, 209.2257690, -404.6186523, 331.2702637, -584.4022217, 613.8444214
4: -186.7911530, 235.2358093, -297.8524780, 368.8239441, -555.6150513, 533.0882568

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780225, upper bound: 141.0779769
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -79.0269623, 66.4630280, -83.3206024, 70.4338150, -149.4607391, 149.7836304
1: -300.7170105, 247.8014984, -316.4634399, 262.3117981, -563.0287476, 564.2649536
2: -161.4815674, 251.7355042, -170.6630859, 267.7605896, -429.2421570, 422.3985901
3: -277.0319214, 226.3745270, -291.4399414, 239.7019501, -516.7337646, 517.8144531
4: -204.2860718, 253.3749542, -214.8449249, 269.0508423, -473.3368530, 468.2197876

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791715, upper bound: 141.0779827
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792470, upper bound: 141.0779920
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -79.0269623, 66.4630280, -115.5957718, 97.4592056, -176.4861603, 182.0588074
1: -300.7170105, 247.8014984, -440.9888916, 363.0383606, -663.7553711, 688.7903442
2: -161.4815674, 251.7355042, -234.1324768, 368.2810364, -529.7625122, 485.8679810
3: -277.0319214, 226.3745270, -404.6186523, 331.2702637, -608.3021851, 630.9931641
4: -204.2860718, 253.3749542, -297.8524780, 368.8239441, -573.1098022, 551.2274170

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791714, upper bound: 141.0779827
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792470, upper bound: 141.0779920
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -72.4293976, 61.4979401, -89.2379761, 74.8676834, -147.2970886, 150.7359161
1: -274.5116272, 229.1019897, -340.0731506, 278.9989929, -553.5105591, 569.1751099
2: -148.8453064, 233.9254303, -182.0450592, 283.5705872, -432.4158325, 415.9704895
3: -253.1319733, 209.2257690, -312.9615784, 254.9642944, -508.0962524, 522.1873779
4: -186.7911530, 235.2358093, -230.6401672, 285.0325623, -471.8236694, 465.8759766

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780335, upper bound: 141.0792250
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0782064, upper bound: 141.0792579
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -72.4293976, 61.4979401, -119.8994751, 100.6422653, -173.0716400, 181.3974152
1: -274.5116272, 229.1019897, -458.4376221, 374.6204834, -649.1319580, 687.5396118
2: -148.8453064, 233.9254303, -242.3918762, 380.4034119, -529.2487183, 476.3172607
3: -253.1319733, 209.2257690, -420.4132385, 342.0752563, -595.2072144, 629.6390381
4: -186.7911530, 235.2358093, -309.3444519, 381.0546265, -567.8457642, 544.5802612

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780225, upper bound: 141.0792250
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0792579
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -79.0269623, 66.4630280, -89.2379761, 74.8676834, -153.8946533, 155.7010040
1: -300.7170105, 247.8014984, -340.0731506, 278.9989929, -579.7160034, 587.8746338
2: -161.4815674, 251.7355042, -182.0450592, 283.5705872, -445.0521240, 433.7805786
3: -277.0319214, 226.3745270, -312.9615784, 254.9642944, -531.9961548, 539.3361206
4: -204.2860718, 253.3749542, -230.6401672, 285.0325623, -489.3185425, 484.0150757

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795478, upper bound: 141.0794551
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795464, upper bound: 141.0794551
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -79.0269623, 66.4630280, -119.8994751, 100.6422653, -179.6691895, 186.3625031
1: -300.7170105, 247.8014984, -458.4376221, 374.6204834, -675.3375244, 706.2391357
2: -161.4815674, 251.7355042, -242.3918762, 380.4034119, -541.8850098, 494.1273804
3: -277.0319214, 226.3745270, -420.4132385, 342.0752563, -619.1071777, 646.7877197
4: -204.2860718, 253.3749542, -309.3444519, 381.0546265, -585.3405762, 562.7194214

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795478, upper bound: 141.0794551
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795464, upper bound: 141.0794551
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -101.6661301, 85.2807693, -84.9776993, 71.2030945, -172.8692017, 170.2584686
1: -388.0832825, 317.1436157, -323.8356323, 264.8934631, -652.9766846, 640.9792480
2: -206.1948853, 321.6758118, -173.4138184, 270.5112915, -476.7061768, 495.0896301
3: -356.8251038, 290.0955811, -297.7329102, 242.4161530, -599.2412720, 587.8284912
4: -262.9872742, 322.8632202, -219.4402771, 271.8333435, -534.8205566, 542.3034668

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788105, upper bound: 141.0790580
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790206, upper bound: 141.0790769
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -101.6661301, 85.2807693, -90.5817871, 75.9953842, -177.6614990, 175.8625488
1: -388.0832825, 317.1436157, -345.2115479, 283.0832214, -671.1664429, 662.3551025
2: -206.1948853, 321.6758118, -184.8361359, 287.5989075, -493.7937622, 506.5119629
3: -356.8251038, 290.0955811, -317.4785461, 258.8067017, -615.6318359, 607.5740356
4: -262.9872742, 322.8632202, -234.0648804, 289.0513000, -552.0385742, 556.9281006

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788105, upper bound: 141.0793668
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790206, upper bound: 141.0793855
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -102.1519775, 85.8980942, -87.1407394, 73.0773239, -175.2293091, 173.0388336
1: -390.1710815, 320.0437622, -332.1969604, 271.9396362, -662.1107178, 652.2406616
2: -207.0178223, 324.2344360, -177.7311859, 277.6297913, -484.6476135, 501.9656067
3: -358.3016968, 292.3731995, -305.3159180, 248.7448730, -607.0465698, 597.6890869
4: -263.8296509, 325.0742188, -224.9303284, 278.9521484, -542.7817383, 550.0045166

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792552, upper bound: 141.0791090
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0790970
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -102.1519775, 85.8980942, -92.7949219, 77.8830338, -180.0350037, 178.6929932
1: -390.1710815, 320.0437622, -353.7388611, 290.1627197, -680.3338013, 673.7825317
2: -207.0178223, 324.2344360, -189.2488403, 294.9390869, -501.9569092, 513.4832764
3: -358.3016968, 292.3731995, -325.2194214, 265.2382812, -623.5399780, 617.5926514
4: -263.8296509, 325.0742188, -239.6724091, 296.3587036, -560.1883545, 564.7466431

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792552, upper bound: 141.0794194
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0794067
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -100.0391083, 84.0942383, -117.4940414, 98.5616989, -198.6007996, 201.5882874
1: -382.0672607, 313.2711792, -449.2413025, 366.9660339, -749.0333252, 762.5124512
2: -202.7186890, 317.4246826, -237.4562531, 372.1501770, -574.8688965, 554.8807983
3: -350.9641418, 286.2120667, -412.0325928, 335.1397095, -686.1038818, 698.2446289
4: -258.4129944, 318.3495178, -303.1804810, 372.8971863, -631.3101196, 621.5300293

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790110, upper bound: 141.0793195
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794490, upper bound: 141.0794383
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -100.0391083, 84.0942383, -119.0891647, 99.7581100, -199.7972107, 203.1833954
1: -382.0672607, 313.2711792, -455.1715088, 371.4352417, -753.5024414, 768.4426270
2: -202.7186890, 317.4246826, -241.1285706, 376.4358215, -579.1545410, 558.5532227
3: -350.9641418, 286.2120667, -417.7422791, 339.1120605, -690.0761719, 703.9543457
4: -258.4129944, 318.3495178, -307.7796631, 377.1218262, -635.5347290, 626.1291504

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790110, upper bound: 141.0793198
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794490, upper bound: 141.0794387
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -101.8747864, 85.5148849, -117.4940414, 98.5616989, -200.4364777, 203.0089264
1: -388.9739380, 318.5489807, -449.2413025, 366.9660339, -755.9399414, 767.7901611
2: -206.9012756, 322.4299622, -237.4562531, 372.1501770, -579.0513916, 559.8861694
3: -357.5431519, 290.9306946, -412.0325928, 335.1397095, -692.6828613, 702.9631958
4: -263.6125793, 323.2859497, -303.1804810, 372.8971863, -636.5097656, 626.4663086

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793801, upper bound: 141.0794263
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -101.8747864, 85.5148849, -119.0891647, 99.7581100, -201.6329041, 204.6040344
1: -388.9739380, 318.5489807, -455.1715088, 371.4352417, -760.4091187, 773.7203369
2: -206.9012756, 322.4299622, -241.1285706, 376.4358215, -583.3370361, 563.5585327
3: -357.5431519, 290.9306946, -417.7422791, 339.1120605, -696.6552124, 708.6728516
4: -263.6125793, 323.2859497, -307.7796631, 377.1218262, -640.7343140, 631.0655518

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793801, upper bound: 141.0794263
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794308
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -99.9498291, 84.5526505, -85.4469833, 72.2266846, -172.1765137, 169.9996185
1: -380.6382446, 315.2061462, -324.5920410, 269.0182800, -649.6564941, 639.7982178
2: -203.3471832, 319.6785278, -174.9865112, 274.5389099, -477.8861084, 494.6650391
3: -350.0321350, 287.5701904, -298.7996826, 245.7848969, -595.8170166, 586.3698730
4: -257.8861084, 320.7027283, -220.2798767, 275.7629089, -533.6489258, 540.9826050

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778361, upper bound: 141.0781742
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0781941
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -99.9498291, 84.5526505, -91.4260101, 76.7240906, -176.6739197, 175.9786377
1: -380.6382446, 315.2061462, -348.4479065, 285.8861389, -666.5243530, 663.6539307
2: -203.3471832, 319.6785278, -186.4906006, 290.5944519, -493.9416504, 506.1690674
3: -350.0321350, 287.5701904, -320.5222168, 261.2350769, -611.2672119, 608.0924072
4: -257.8861084, 320.7027283, -236.2332611, 291.9217834, -549.8078613, 556.9359741

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778361, upper bound: 141.0792152
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0792470
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -104.6050720, 87.9910889, -86.0617676, 72.1637802, -176.7688599, 174.0528107
1: -399.4220886, 328.0237427, -328.0349426, 268.5652771, -667.9871826, 656.0586548
2: -212.1596222, 331.9995728, -175.5608215, 274.1769409, -486.3365479, 507.5603638
3: -367.0649414, 299.3190308, -301.5751038, 245.6044922, -612.6692505, 600.8940430
4: -270.3831787, 332.9727783, -222.1788635, 275.4733887, -545.8565674, 555.1516113

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793198, upper bound: 141.0794423
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793451, upper bound: 141.0794394
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -104.6050720, 87.9910889, -91.9441833, 77.1640167, -181.7690887, 179.9352570
1: -399.4220886, 328.0237427, -350.4537048, 287.5057068, -686.9277954, 678.4773560
2: -212.1596222, 331.9995728, -187.5400238, 292.2395935, -504.3991394, 519.5395508
3: -367.0649414, 299.3190308, -322.2946777, 262.7582703, -629.8230591, 621.6137085
4: -270.3831787, 332.9727783, -237.5317688, 293.6094666, -563.9926758, 570.5045166

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793198, upper bound: 141.0794495
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793451, upper bound: 141.0794417
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -99.9498291, 84.5526505, -115.5957718, 97.4592056, -197.4090271, 200.1483917
1: -380.6382446, 315.2061462, -440.9888916, 363.0383606, -743.6766357, 756.1950684
2: -203.3471832, 319.6785278, -234.1324768, 368.2810364, -571.6281738, 553.8109131
3: -350.0321350, 287.5701904, -404.6186523, 331.2702637, -681.3023682, 692.1888428
4: -257.8861084, 320.7027283, -297.8524780, 368.8239441, -626.7098389, 618.5551758

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778322, upper bound: 141.0779697
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -99.9498291, 84.5526505, -119.8994751, 100.6422653, -200.5920868, 204.4521179
1: -380.6382446, 315.2061462, -458.4376221, 374.6204834, -755.2586670, 773.6437988
2: -203.3471832, 319.6785278, -242.3918762, 380.4034119, -583.7506104, 562.0703125
3: -350.0321350, 287.5701904, -420.4132385, 342.0752563, -692.1074219, 707.9833984
4: -257.8861084, 320.7027283, -309.3444519, 381.0546265, -638.9406128, 630.0471802

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -102.2181015, 85.9636612, -120.8057327, 101.3485413, -203.5666199, 206.7693939
1: -390.0709839, 320.4891052, -461.5892029, 376.8259583, -766.8969116, 782.0783081
2: -207.4062805, 324.1446838, -244.3948212, 382.2101135, -589.6163940, 568.5394287
3: -358.6545105, 292.4656067, -423.6982422, 344.3621826, -703.0166016, 716.1637573
4: -264.3428345, 325.3729858, -312.0485535, 383.0444336, -647.3872681, 637.4215088

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793378
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793378
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -104.4519730, 87.8661346, -120.6801605, 101.2934723, -205.7454529, 208.5462952
1: -398.8620300, 327.5523376, -461.4552917, 377.0657959, -775.9275513, 789.0075073
2: -211.8612366, 331.5173035, -243.9089050, 382.8167725, -594.6779785, 575.4262085
3: -366.5505371, 298.8848572, -423.0960388, 344.3064270, -710.8567505, 721.9808960
4: -270.0058899, 332.4860840, -311.3212280, 383.4758301, -653.4815674, 643.8072510

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793674
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793674
time: 0.98 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.86 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0791541
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794806, upper bound: 141.0792373
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0794330
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794806, upper bound: 141.0795182
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793966, upper bound: 141.0793016
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794387, upper bound: 141.0793059
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0790642, upper bound: 141.0792186
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794324, upper bound: 141.0793029
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0780225, upper bound: 141.0779769
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0780225, upper bound: 141.0779769
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0779919
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0791715, upper bound: 141.0779827
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0792470, upper bound: 141.0779920
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0791714, upper bound: 141.0779827
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0792470, upper bound: 141.0779920
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0780335, upper bound: 141.0792250
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0782064, upper bound: 141.0792579
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0780225, upper bound: 141.0792250
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0781941, upper bound: 141.0792579
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0795478, upper bound: 141.0794551
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0795464, upper bound: 141.0794551
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0795478, upper bound: 141.0794551
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0795464, upper bound: 141.0794551
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0788105, upper bound: 141.0790580
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0790206, upper bound: 141.0790769
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0788105, upper bound: 141.0793668
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0790206, upper bound: 141.0793855
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0792552, upper bound: 141.0791090
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0790970
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0792552, upper bound: 141.0794194
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0794067
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0790110, upper bound: 141.0793195
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794490, upper bound: 141.0794383
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0790110, upper bound: 141.0793198
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794490, upper bound: 141.0794387
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793801, upper bound: 141.0794263
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0794070, upper bound: 141.0794308
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0778361, upper bound: 141.0781742
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0781941
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0778361, upper bound: 141.0792152
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0779919, upper bound: 141.0792470
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793198, upper bound: 141.0794423
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793451, upper bound: 141.0794394
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793198, upper bound: 141.0794495
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793451, upper bound: 141.0794417
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0778322, upper bound: 141.0779697
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0779874, upper bound: 141.0779874
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793378
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793378
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793674
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793674

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -70.0483627, 58.8776131, -83.3814468, 69.8835449, -139.9319153, 142.2590027
1: -266.7187805, 218.9536896, -317.8380127, 259.9370728, -526.6557617, 536.7915649
2: -143.2226105, 223.7831573, -170.0516510, 265.3237915, -408.5463867, 393.8348083
3: -245.9184723, 200.3550415, -292.2760010, 237.9141846, -483.8326416, 492.6310120
4: -181.2534637, 224.9312439, -215.2946930, 266.7060547, -447.9594727, 440.2259521

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0791541
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0791541
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -71.7348404, 60.2571869, -82.2376099, 68.9272537, -140.6620941, 142.4947968
1: -272.9916077, 223.6447144, -313.4465332, 256.4143372, -529.4059448, 537.0912476
2: -146.7405090, 229.6200867, -167.7726288, 261.6750793, -408.4155884, 397.3927002
3: -251.8039703, 204.6659088, -288.2765808, 234.6711121, -486.4750977, 492.9423523
4: -185.3827515, 230.8279419, -212.3535461, 263.0523682, -448.4350586, 443.1814270

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794806, upper bound: 141.0792373
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0792373
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -70.0483627, 58.8776131, -89.0542297, 74.6992340, -144.7475891, 147.9318390
1: -266.7187805, 218.9536896, -339.4610596, 278.2731934, -544.9916992, 558.4146729
2: -143.2226105, 223.7831573, -181.6340485, 282.9969788, -426.2195740, 405.4172058
3: -245.9184723, 200.3550415, -312.2731628, 254.4270020, -500.3454590, 512.6281128
4: -181.2534637, 224.9312439, -230.1050873, 284.5162048, -465.7696533, 455.0363159

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791652, upper bound: 141.0794330
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792166, upper bound: 141.0794330
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -71.7348404, 60.2571869, -87.8863678, 73.7333450, -145.4681854, 148.1435242
1: -272.9916077, 223.6447144, -334.9754333, 274.7316589, -547.7232666, 558.6201172
2: -146.7405090, 229.6200867, -179.2984619, 279.3880310, -426.1285400, 408.9185486
3: -251.8039703, 204.6659088, -308.1809692, 251.1555786, -502.9595337, 512.8468018
4: -185.3827515, 230.8279419, -227.0876160, 280.9006653, -466.2833557, 457.9155273

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795357, upper bound: 141.0795182
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792166, upper bound: 141.0795182
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -68.4785080, 57.5514679, -118.2168579, 99.1221466, -167.6006165, 175.7683105
1: -260.6252747, 214.1645966, -451.6958313, 368.5388794, -629.1641235, 665.8604126
2: -139.9373932, 218.4337311, -239.1321411, 373.4939880, -513.4313965, 457.5658569
3: -240.3505859, 195.9859467, -414.7155762, 336.8482971, -577.1988525, 610.7013550
4: -177.1854553, 219.7263184, -305.4321289, 374.4116516, -551.5970459, 525.1583862

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793429, upper bound: 141.0790501
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793429, upper bound: 141.0793016
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -68.3153152, 57.3884010, -118.2168579, 99.1221466, -167.4374542, 175.6052551
1: -260.1502075, 213.5570679, -451.6958313, 368.5388794, -628.6889648, 665.2528687
2: -139.5573425, 217.7882690, -239.1321411, 373.4939880, -513.0513306, 456.9203796
3: -239.8479462, 195.4268188, -414.7155762, 336.8482971, -576.6962280, 610.1423950
4: -176.7701111, 219.0273743, -305.4321289, 374.4116516, -551.1817017, 524.4594727

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793855, upper bound: 141.0790540
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793855, upper bound: 141.0793059
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -69.8480835, 58.7150497, -117.8385162, 98.8542252, -168.7023010, 176.5535126
1: -265.9314270, 218.3477936, -450.5674133, 368.0227661, -633.9540405, 668.9152222
2: -142.8386230, 223.1789551, -238.1504059, 373.2743530, -516.1129761, 461.3293457
3: -245.2144775, 199.7938080, -413.2059937, 336.1080322, -581.3224487, 612.9997559
4: -180.7448425, 224.3259888, -304.0618286, 374.0274353, -554.7722168, 528.3878174

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790382, upper bound: 141.0790861
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790382, upper bound: 141.0792186
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -71.6020432, 60.1464539, -116.5615387, 97.7840881, -169.3861389, 176.7079926
1: -272.4830322, 223.2280731, -445.6526489, 364.1179810, -636.6010132, 668.8807373
2: -146.4713440, 229.2003021, -235.6277313, 369.2354431, -515.7067871, 464.8279724
3: -251.3380280, 204.2876434, -408.7679749, 332.5215454, -583.8594360, 613.0556030
4: -185.0375061, 230.4076996, -300.7936096, 369.9897156, -555.0272217, 531.2012939

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794035, upper bound: 141.0791764
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794035, upper bound: 141.0793029
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -71.9924850, 61.1309776, -83.3206024, 70.4338150, -142.4263000, 144.4515686
1: -272.8479614, 227.7306213, -316.4634399, 262.3117981, -535.1597290, 544.1940308
2: -147.9460297, 232.4934692, -170.6630859, 267.7605896, -415.7066040, 403.1565247
3: -251.6048584, 207.9764404, -291.4399414, 239.7019501, -491.3067322, 499.4163818
4: -185.6513977, 233.8101196, -214.8449249, 269.0508423, -454.7022095, 448.6549683

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780150, upper bound: 141.0780150
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780150, upper bound: 141.0781868
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -71.8802185, 60.9993286, -83.3206024, 70.4338150, -142.3140259, 144.3199310
1: -272.5813599, 227.3026581, -316.4634399, 262.3117981, -534.8931274, 543.7659912
2: -147.6000824, 231.8850555, -170.6630859, 267.7605896, -415.3605957, 402.5480652
3: -251.2694092, 207.5662842, -291.4399414, 239.7019501, -490.9712524, 499.0062256
4: -185.3891296, 233.1933594, -214.8449249, 269.0508423, -454.4399719, 448.0382690

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781868, upper bound: 141.0780332
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781868, upper bound: 141.0782049
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -71.9924850, 61.1309776, -115.5957718, 97.4592056, -169.4516907, 176.7267456
1: -272.8479614, 227.7306213, -440.9888916, 363.0383606, -635.8863525, 668.7194824
2: -147.9460297, 232.4934692, -234.1324768, 368.2810364, -516.2270508, 466.6259155
3: -251.6048584, 207.9764404, -404.6186523, 331.2702637, -582.8751221, 612.5950317
4: -185.6513977, 233.8101196, -297.8524780, 368.8239441, -554.4753418, 531.6625977

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780025, upper bound: 141.0778218
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780025, upper bound: 141.0779769
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -71.8802185, 60.9993286, -115.5957718, 97.4592056, -169.3394165, 176.5950928
1: -272.5813599, 227.3026581, -440.9888916, 363.0383606, -635.6197510, 668.2914429
2: -147.6000824, 231.8850555, -234.1324768, 368.2810364, -515.8811035, 466.0174866
3: -251.2694092, 207.5662842, -404.6186523, 331.2702637, -582.5396729, 612.1849365
4: -185.3891296, 233.1933594, -297.8524780, 368.8239441, -554.2130737, 531.0458374

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781742, upper bound: 141.0778362
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781742, upper bound: 141.0779919
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -78.6253357, 66.1257248, -83.3206024, 70.4338150, -149.0591278, 149.4463043
1: -299.1929321, 246.5452423, -316.4634399, 262.3117981, -561.5045166, 563.0086670
2: -160.6513214, 250.4285431, -170.6630859, 267.7605896, -428.4119263, 421.0916138
3: -275.6244812, 225.2324829, -291.4399414, 239.7019501, -515.3264160, 516.6724243
4: -203.2362061, 252.0717316, -214.8449249, 269.0508423, -472.2870178, 466.9166260

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792334, upper bound: 141.0780319
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792334, upper bound: 141.0782048
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -78.3810425, 65.8988571, -83.3206024, 70.4338150, -148.8148499, 149.2194214
1: -298.4010925, 245.7524109, -316.4634399, 262.3117981, -560.7128296, 562.2158203
2: -160.0608368, 249.4645538, -170.6630859, 267.7605896, -427.8214111, 420.1275940
3: -274.8000183, 224.5046539, -291.4399414, 239.7019501, -514.5018921, 515.9445801
4: -202.6246948, 251.1040955, -214.8449249, 269.0508423, -471.6755066, 465.9489746

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781868, upper bound: 141.0780332
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781868, upper bound: 141.0782050
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -78.6253357, 66.1257248, -115.5957718, 97.4592056, -176.0845337, 181.7214661
1: -299.1929321, 246.5452423, -440.9888916, 363.0383606, -662.2312012, 687.5340576
2: -160.6513214, 250.4285431, -234.1324768, 368.2810364, -528.9323120, 484.5610352
3: -275.6244812, 225.2324829, -404.6186523, 331.2702637, -606.8947754, 629.8510742
4: -203.2362061, 252.0717316, -297.8524780, 368.8239441, -572.0601196, 549.9241943

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791369, upper bound: 141.0778268
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791369, upper bound: 141.0779827
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -78.3810425, 65.8988571, -115.5957718, 97.4592056, -175.8402405, 181.4945831
1: -298.4010925, 245.7524109, -440.9888916, 363.0383606, -661.4394531, 686.7412109
2: -160.0608368, 249.4645538, -234.1324768, 368.2810364, -528.3417969, 483.5969849
3: -274.8000183, 224.5046539, -404.6186523, 331.2702637, -606.0703125, 629.1232910
4: -202.6246948, 251.1040955, -297.8524780, 368.8239441, -571.4485474, 548.9565430

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781742, upper bound: 141.0778361
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781742, upper bound: 141.0779920
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -71.9924850, 61.1309776, -89.2379761, 74.8676834, -146.8601685, 150.3689423
1: -272.8479614, 227.7306213, -340.0731506, 278.9989929, -551.8469238, 567.8037720
2: -147.9460297, 232.4934692, -182.0450592, 283.5705872, -431.5166016, 414.5385132
3: -251.6048584, 207.9764404, -312.9615784, 254.9642944, -506.5691528, 520.9379883
4: -185.6513977, 233.8101196, -230.6401672, 285.0325623, -470.6838989, 464.4501953

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780319, upper bound: 141.0792334
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780319, upper bound: 141.0793162
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -71.8802185, 60.9993286, -89.2379761, 74.8676834, -146.7478943, 150.2373047
1: -272.5813599, 227.3026581, -340.0731506, 278.9989929, -551.5803223, 567.3756714
2: -147.6000824, 231.8850555, -182.0450592, 283.5705872, -431.1705627, 413.9300842
3: -251.2694092, 207.5662842, -312.9615784, 254.9642944, -506.2337036, 520.5278320
4: -185.3891296, 233.1933594, -230.6401672, 285.0325623, -470.4216919, 463.8335266

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0782048, upper bound: 141.0792694
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0782048, upper bound: 141.0793514
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -71.9924850, 61.1309776, -119.8994751, 100.6422653, -172.6347504, 181.0304565
1: -272.8479614, 227.7306213, -458.4376221, 374.6204834, -647.4683838, 686.1682129
2: -147.9460297, 232.4934692, -242.3918762, 380.4034119, -528.3493652, 474.8852844
3: -251.6048584, 207.9764404, -420.4132385, 342.0752563, -593.6800537, 628.3895874
4: -185.6513977, 233.8101196, -309.3444519, 381.0546265, -566.7060547, 543.1545410

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780263, upper bound: 141.0791539
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0780263, upper bound: 141.0792250
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -71.8802185, 60.9993286, -119.8994751, 100.6422653, -172.5224762, 180.8988037
1: -272.5813599, 227.3026581, -458.4376221, 374.6204834, -647.2017822, 685.7402344
2: -147.6000824, 231.8850555, -242.3918762, 380.4034119, -528.0034790, 474.2768555
3: -251.2694092, 207.5662842, -420.4132385, 342.0752563, -593.3446655, 627.9794922
4: -185.3891296, 233.1933594, -309.3444519, 381.0546265, -566.4437256, 542.5378418

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781992, upper bound: 141.0791918
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0781742, upper bound: 141.0792579
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -76.7241745, 64.5193253, -87.7941742, 73.6457367, -150.3699036, 152.3134918
1: -291.9664307, 240.4738312, -334.5727234, 274.4237366, -566.3900757, 575.0465698
2: -156.7766876, 244.3218842, -179.1001282, 278.9746094, -435.7512817, 423.4219666
3: -269.0336609, 219.6940918, -307.9447632, 250.7857056, -519.8193359, 527.6388550
4: -198.3420563, 245.9557648, -226.9405060, 280.4547729, -478.7968140, 472.8962708

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794888, upper bound: 141.0792444
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794888, upper bound: 141.0795397
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -80.5573883, 67.7083435, -88.3193283, 74.0894699, -154.6468353, 156.0276794
1: -306.5738525, 252.4504852, -336.5563354, 276.1100769, -582.6838989, 589.0067139
2: -164.7571259, 255.7149811, -180.2200775, 280.4876709, -445.2448120, 435.9349976
3: -282.6467590, 230.6044006, -309.7534180, 252.3257751, -534.9725342, 540.3577271
4: -208.7987671, 257.4158936, -228.2871246, 281.9192200, -490.7179871, 485.7029419

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792991, upper bound: 141.0792421
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792991, upper bound: 141.0795397
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -76.7241745, 64.5193253, -118.5319366, 99.4695587, -176.1937103, 183.0512543
1: -291.9664307, 240.4738312, -453.2084961, 370.2276306, -662.1940918, 693.6822510
2: -156.7766876, 244.3218842, -239.6228943, 375.9266357, -532.7033081, 483.9447632
3: -269.0336609, 219.6940918, -415.6744995, 338.0625000, -607.0961914, 635.3685303
4: -198.3420563, 245.9557648, -305.8482056, 376.5677490, -574.9096680, 551.8039551

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794458, upper bound: 141.0793229
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795128, upper bound: 141.0794029
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795128, upper bound: 141.0794551
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -80.5573883, 67.7083435, -118.9281387, 99.8149414, -180.3723145, 186.6364746
1: -306.5738525, 252.4504852, -454.7333374, 371.5054932, -678.0793457, 707.1838379
2: -164.7571259, 255.7149811, -240.4000244, 377.3235779, -542.0806885, 496.1149292
3: -282.6467590, 230.6044006, -417.0205994, 339.2004395, -621.8471680, 647.6250000
4: -208.7987671, 257.4158936, -306.8370361, 377.9255066, -586.7242432, 564.2526855

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794446, upper bound: 141.0793312
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794356, upper bound: 141.0793293
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -99.2272491, 83.1808090, -83.5502930, 69.9799728, -169.2071991, 166.7310791
1: -378.7477722, 309.3231812, -318.3884277, 260.3456421, -639.0931396, 627.7114868
2: -201.2533264, 313.7877197, -170.5088959, 265.7843323, -467.0376282, 484.2966003
3: -348.3375854, 282.9581604, -292.7705383, 238.2659760, -586.6035156, 575.7286987
4: -256.7364197, 315.0253906, -215.7815552, 267.0921936, -523.8286133, 530.8068848

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788105, upper bound: 141.0790580
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788105, upper bound: 141.0790580
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -101.0965881, 84.6502686, -84.0849762, 70.4414673, -171.5380554, 168.7352142
1: -385.7569885, 314.9193420, -320.4382019, 262.1016235, -647.8585815, 635.3575439
2: -205.4685822, 318.9314575, -171.6352997, 267.6577454, -473.1263428, 490.5667114
3: -354.9864502, 287.9582214, -294.6216431, 239.8543091, -594.8406372, 582.5797729
4: -262.0045166, 320.1433716, -217.1457214, 268.9444580, -530.9489746, 537.2889404

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790206, upper bound: 141.0790769
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790206, upper bound: 141.0790769
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -99.2272491, 83.1808090, -89.1322479, 74.7674332, -173.9946899, 172.3130493
1: -378.7477722, 309.3231812, -339.6772766, 278.4976807, -657.2452393, 649.0003052
2: -201.2533264, 313.7877197, -181.8911743, 282.9699707, -484.2232971, 495.6788940
3: -348.3375854, 282.9581604, -312.4388428, 254.6220703, -602.9595947, 595.3969727
4: -256.7364197, 315.0253906, -230.3508759, 284.4755249, -541.2119141, 545.3760986

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788877, upper bound: 141.0792999
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788877, upper bound: 141.0793668
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -101.0965881, 84.6502686, -89.6905060, 75.2379608, -176.3345490, 174.3407593
1: -385.7569885, 314.9193420, -341.7973938, 280.2722473, -666.0292358, 656.7166748
2: -205.4685822, 318.9314575, -183.0641174, 284.7100220, -490.1785889, 501.9955139
3: -354.9864502, 287.9582214, -314.3641052, 256.2307434, -611.2171631, 602.3222656
4: -262.0045166, 320.1433716, -231.7806244, 286.0090332, -548.0135498, 551.9238892

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790981, upper bound: 141.0793194
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790981, upper bound: 141.0793855
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -99.6693344, 83.7816391, -85.7444763, 71.8833466, -171.5526733, 169.5261078
1: -380.6648865, 312.0918579, -326.8735046, 267.4553528, -648.1202393, 638.9652710
2: -201.9722595, 316.2343445, -174.8867645, 273.0067139, -474.9788513, 491.1210632
3: -349.6667175, 285.1222534, -300.4658508, 244.6730347, -594.3397217, 585.5881348
4: -257.4614868, 317.1522217, -221.3511353, 274.3143005, -531.7757568, 538.5033569

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0790970
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0790970
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -101.4333649, 85.1412048, -86.2220764, 72.2950058, -173.7283478, 171.3632812
1: -387.2905579, 317.1501770, -328.7004089, 269.0450134, -656.3355713, 645.8504639
2: -206.0099335, 321.0086060, -175.8988342, 274.6768188, -480.6867371, 496.9074402
3: -355.9932861, 289.6412964, -302.1109314, 246.0941162, -602.0874023, 591.7521362
4: -262.4812012, 321.8669434, -222.5682678, 275.9722290, -538.4534302, 544.4351807

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0790970
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791491, upper bound: 141.0790970
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -99.6693344, 83.7816391, -91.3591919, 76.6652985, -176.3346252, 175.1408386
1: -380.6648865, 312.0918579, -348.2592468, 285.6152954, -666.2800903, 660.3510742
2: -201.9722595, 316.2343445, -186.3320312, 290.3768921, -492.3490601, 502.5663757
3: -349.6667175, 285.1222534, -320.2291870, 261.0853882, -610.7520752, 605.3514404
4: -257.4614868, 317.1522217, -235.9931641, 291.8147583, -549.2760620, 553.1453857

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793256, upper bound: 141.0793519
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793256, upper bound: 141.0794007
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -101.4333649, 85.1412048, -91.8821411, 77.1081238, -178.5414886, 177.0233307
1: -387.2905579, 317.1501770, -350.2440491, 287.2861938, -674.5767822, 667.3941040
2: -206.0099335, 321.0086060, -187.4308472, 291.9484863, -497.9584045, 508.4394226
3: -355.9932861, 289.6412964, -322.0292358, 262.6033630, -618.5966797, 611.6704712
4: -262.4812012, 321.8669434, -237.3312073, 293.2467957, -555.7280273, 559.1981201

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792270, upper bound: 141.0793395
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792270, upper bound: 141.0793679
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -99.3707733, 83.4913483, -117.2376785, 98.3348770, -197.7056580, 200.7290039
1: -379.5181885, 310.9216003, -448.2673645, 366.0671387, -745.5853271, 759.1889648
2: -201.3567352, 315.2721558, -236.9281158, 371.3081360, -572.6648560, 552.2002563
3: -348.6470337, 284.1423035, -411.1422119, 334.3367310, -682.9837646, 695.2845459
4: -256.6945190, 316.1800842, -302.5239258, 372.0507202, -628.7452393, 618.7039795

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0787611, upper bound: 141.0792369
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0787876, upper bound: 141.0792787
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -99.1609192, 83.2325821, -116.0015106, 97.2982101, -196.4591370, 199.2340851
1: -378.7974243, 310.0280762, -443.5122681, 362.2886353, -741.0859985, 753.5403442
2: -201.0260468, 314.9085083, -234.4850006, 367.4138489, -568.4398804, 549.3934937
3: -347.9317017, 283.2778931, -406.8491516, 330.8652649, -678.7969971, 690.1270752
4: -256.0206909, 315.8699341, -299.3591309, 368.1494751, -624.1701050, 615.2290649

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0787611, upper bound: 141.0794002
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794490, upper bound: 141.0794397
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -99.3707733, 83.4913483, -118.8270721, 99.5253983, -198.8961792, 202.3184204
1: -379.5181885, 310.9216003, -454.1757812, 370.5166016, -750.0347900, 765.0974121
2: -201.3567352, 315.2721558, -240.5883179, 375.5709229, -576.9276733, 555.8604126
3: -348.6470337, 284.1423035, -416.8316040, 338.2929688, -686.9399414, 700.9738770
4: -256.6945190, 316.1800842, -307.1080627, 376.2538452, -632.9483643, 623.2881470

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0787088, upper bound: 141.0792278
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0787196, upper bound: 141.0792658
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -99.1609192, 83.2325821, -117.5887527, 98.4907684, -197.6516571, 200.8213348
1: -378.7974243, 310.0280762, -449.4071960, 366.7476196, -745.5449219, 759.4351807
2: -201.0260468, 314.9085083, -238.1477661, 371.6890564, -572.7150879, 553.0562744
3: -347.9317017, 283.2778931, -412.5173035, 334.8408203, -682.7725220, 695.7951660
4: -256.0206909, 315.8699341, -303.9311829, 372.3663940, -628.3870850, 619.8011475

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0787088, upper bound: 141.0793978
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794700, upper bound: 141.0794387
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -101.3487244, 85.0736771, -119.0891647, 99.7581100, -201.1068420, 204.1628265
1: -386.9527283, 316.9262390, -455.1715088, 371.4352417, -758.3877563, 772.0977783
2: -205.8134613, 320.7004089, -241.1285706, 376.4358215, -582.2492676, 561.8289795
3: -355.6791077, 289.4538879, -417.7422791, 339.1120605, -694.7911377, 707.1959839
4: -262.2547607, 321.5689697, -307.7796631, 377.1218262, -639.3765259, 629.3486328

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793755, upper bound: 141.0793856
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -101.1966705, 84.9074631, -119.0891647, 99.7581100, -200.9547729, 203.9966278
1: -386.4968262, 316.3063660, -455.1715088, 371.4352417, -757.9320068, 771.4778442
2: -205.4338379, 320.0925598, -241.1285706, 376.4358215, -581.8696289, 561.2211304
3: -355.2050476, 288.8860474, -417.7422791, 339.1120605, -694.3170776, 706.6280518
4: -261.8527222, 320.9227600, -307.7796631, 377.1218262, -638.9744873, 628.7023926

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794024, upper bound: 141.0793900
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794024, upper bound: 141.0794308
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -99.4408340, 84.1175079, -85.4469833, 72.2266846, -171.6675110, 169.5644836
1: -378.6616211, 313.6019592, -324.5920410, 269.0182800, -647.6799316, 638.1939697
2: -202.2935486, 317.9903259, -174.9865112, 274.5389099, -476.8324585, 492.9768372
3: -348.2373657, 286.1067810, -298.7996826, 245.7848969, -594.0222778, 584.9064941
4: -256.5867004, 319.0419922, -220.2798767, 275.7629089, -532.3495483, 539.3218994

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778218, upper bound: 141.0780025
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778218, upper bound: 141.0781742
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -99.3666382, 84.0251465, -85.4469833, 72.2266846, -171.5932922, 169.4721375
1: -378.5270996, 313.2450256, -324.5920410, 269.0182800, -647.5453491, 637.8370361
2: -202.0617676, 317.6059570, -174.9865112, 274.5389099, -476.6006775, 492.5924683
3: -348.0311584, 285.7840881, -298.7996826, 245.7848969, -593.8159790, 584.5837402
4: -256.3805542, 318.5953674, -220.2798767, 275.7629089, -532.1433716, 538.8752441

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779769, upper bound: 141.0780225
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779769, upper bound: 141.0781941
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -99.4408340, 84.1175079, -91.4260101, 76.7240906, -176.1649017, 175.5435028
1: -378.6616211, 313.6019592, -348.4479065, 285.8861389, -664.5477295, 662.0498657
2: -202.2935486, 317.9903259, -186.4906006, 290.5944519, -492.8880005, 504.4808960
3: -348.2373657, 286.1067810, -320.5222168, 261.2350769, -609.4722900, 606.6290283
4: -256.5867004, 319.0419922, -236.2332611, 291.9217834, -548.5084839, 555.2752686

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778268, upper bound: 141.0791369
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778268, upper bound: 141.0792152
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -99.3666382, 84.0251465, -91.4260101, 76.7240906, -176.0906830, 175.4511566
1: -378.5270996, 313.2450256, -348.4479065, 285.8861389, -664.4132080, 661.6927490
2: -202.0617676, 317.6059570, -186.4906006, 290.5944519, -492.6562195, 504.0964966
3: -348.0311584, 285.7840881, -320.5222168, 261.2350769, -609.2659912, 606.3062744
4: -256.3805542, 318.5953674, -236.2332611, 291.9217834, -548.3023682, 554.8286133

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779827, upper bound: 141.0791714
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779769, upper bound: 141.0792470
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -103.5991745, 86.9543839, -83.7578659, 70.1711807, -173.7703094, 170.7122498
1: -395.3953552, 323.5069580, -319.1353760, 261.1228638, -656.5181885, 642.6423340
2: -210.3790283, 327.8496094, -170.9649048, 266.6144714, -476.9934998, 498.8144226
3: -363.8281250, 295.6271362, -293.4977417, 238.9226532, -602.7507935, 589.1248779
4: -268.2227173, 329.1765442, -216.3267517, 267.9157410, -536.1381836, 545.5032959

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793059, upper bound: 141.0794387
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792707, upper bound: 141.0791460
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -104.2032166, 87.6630020, -85.9202118, 72.0475082, -176.2506866, 173.5831757
1: -397.9522095, 326.7857971, -327.4866028, 268.1311035, -666.0833130, 654.2723999
2: -211.3761292, 330.7336121, -175.2794342, 273.7441101, -485.1202087, 506.0130615
3: -365.7140198, 298.1788940, -301.0744629, 245.2118530, -610.9257202, 599.2533569
4: -269.3924561, 331.6949463, -221.8128815, 275.0419922, -544.4342651, 553.5077515

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793072, upper bound: 141.0794358
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792720, upper bound: 141.0791273
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -103.5991745, 86.9543839, -89.5850754, 75.1557159, -178.7548523, 176.5394592
1: -395.3953552, 323.5069580, -341.3619080, 279.9804382, -675.3755493, 664.8686523
2: -210.3790283, 327.8496094, -182.8354187, 284.4017334, -494.7807617, 510.6850281
3: -363.8281250, 295.6271362, -314.0381775, 255.9224548, -619.7506104, 609.6652222
4: -268.2227173, 329.1765442, -231.5514069, 285.8569031, -554.0794678, 560.7279053

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793378
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0794417
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -104.2032166, 87.6630020, -91.7936707, 77.0400391, -181.2432556, 179.4566345
1: -397.9522095, 326.7857971, -349.8692932, 287.0444641, -684.9967041, 676.6550903
2: -211.3761292, 330.7336121, -187.2422638, 291.7802124, -503.1562805, 517.9758911
3: -365.7140198, 298.1788940, -321.7623901, 262.3393250, -628.0532837, 619.9412842
4: -269.3924561, 331.6949463, -237.1438446, 293.1534119, -562.5456543, 568.8387451

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0793378
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793378, upper bound: 141.0794417
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -99.4408340, 84.1175079, -115.5957718, 97.4592056, -196.9000397, 199.7132721
1: -378.6616211, 313.6019592, -440.9888916, 363.0383606, -741.6999512, 754.5908203
2: -202.2935486, 317.9903259, -234.1324768, 368.2810364, -570.5745850, 552.1227417
3: -348.2373657, 286.1067810, -404.6186523, 331.2702637, -679.5076294, 690.7254028
4: -256.5867004, 319.0419922, -297.8524780, 368.8239441, -625.4105225, 616.8944702

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778143, upper bound: 141.0778143
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0778143, upper bound: 141.0779697
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -99.3666382, 84.0251465, -115.5957718, 97.4592056, -196.8257904, 199.6208954
1: -378.5270996, 313.2450256, -440.9888916, 363.0383606, -741.5654297, 754.2338867
2: -202.0617676, 317.6059570, -234.1324768, 368.2810364, -570.3427124, 551.7384033
3: -348.0311584, 285.7840881, -404.6186523, 331.2702637, -679.3013916, 690.4027100
4: -256.3805542, 318.5953674, -297.8524780, 368.8239441, -625.2044067, 616.4478760

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779697, upper bound: 141.0778322
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0779697, upper bound: 141.0779874
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -103.5991745, 86.9543839, -120.8057327, 101.3485413, -204.9476776, 207.7601166
1: -395.3953552, 323.5069580, -461.5892029, 376.8259583, -772.2212524, 785.0961304
2: -210.3790283, 327.8496094, -244.3948212, 382.2101135, -592.5890503, 572.2442627
3: -363.8281250, 295.6271362, -423.6982422, 344.3621826, -708.1903076, 719.3252563
4: -268.2227173, 329.1765442, -312.0485535, 383.0444336, -651.2670288, 641.2250977

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791505, upper bound: 141.0791001
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791505, upper bound: 141.0793378
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -104.2032166, 87.6630020, -120.8057327, 101.3485413, -205.5517578, 208.4687347
1: -397.9522095, 326.7857971, -461.5892029, 376.8259583, -774.7781982, 788.3750000
2: -211.3761292, 330.7336121, -244.3948212, 382.2101135, -593.5861816, 575.1283569
3: -365.7140198, 298.1788940, -423.6982422, 344.3621826, -710.0761719, 721.8770752
4: -269.3924561, 331.6949463, -312.0485535, 383.0444336, -652.4367676, 643.7434082

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791505, upper bound: 141.0791001
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791505, upper bound: 141.0793378
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -103.5991745, 86.9543839, -120.6801605, 101.2934723, -204.8926086, 207.6345520
1: -395.3953552, 323.5069580, -461.4552917, 377.0657959, -772.4608765, 784.9620972
2: -210.3790283, 327.8496094, -243.9089050, 382.8167725, -593.1958008, 571.7584839
3: -363.8281250, 295.6271362, -423.0960388, 344.3064270, -708.1345215, 718.7231445
4: -268.2227173, 329.1765442, -311.3212280, 383.4758301, -651.6983032, 640.4978027

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791505, upper bound: 141.0793643
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791505, upper bound: 141.0793674
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -104.2032166, 87.6630020, -120.6801605, 101.2934723, -205.4966888, 208.3431244
1: -397.9522095, 326.7857971, -461.4552917, 377.0657959, -775.0178223, 788.2410889
2: -211.3761292, 330.7336121, -243.9089050, 382.8167725, -594.1928711, 574.6425171
3: -365.7140198, 298.1788940, -423.0960388, 344.3064270, -710.0202637, 721.2749023
4: -269.3924561, 331.6949463, -311.3212280, 383.4758301, -652.8680420, 643.0160522

Time for backsubstitution: 1.21 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.13 + 417.02 = 420.15 seconds
