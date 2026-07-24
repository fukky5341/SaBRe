## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 607.026092835328


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574)
1: (-264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331)
2: (-213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147)
3: (-300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574)
4: (-274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 2.39 = 3.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -607.0382336, upper bound: 607.0382336

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338287, upper bound: 607.0375985
time: 0.82 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338198, upper bound: 607.0338198
time: 0.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.89 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 0, lower bound: -607.0338287, upper bound: 607.0375985
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 0, lower bound: -607.0338198, upper bound: 607.0338198

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -298.4826050, 350.0044556, -317.7158813, 374.1755066, -672.6580811, 667.7203369
1: -248.2315521, 282.9363708, -264.8054199, 302.6636658, -550.8950195, 547.7416382
2: -199.9671478, 278.1452026, -213.3653564, 297.9397583, -497.9069214, 491.5105591
3: -281.0348511, 353.4127197, -300.8116760, 377.7047119, -658.7394409, 654.2242432
4: -257.1254883, 376.2516174, -274.6094055, 403.1719055, -660.2973633, 650.8609619

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338198, upper bound: 607.0338198
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338198, upper bound: 607.0338198
time: 0.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -490.6899719, 574.6386108, -315.7005615, 371.7794800, -861.4858398, 890.3391724
1: -414.5460510, 463.7775879, -263.1916199, 300.7552490, -712.8823853, 726.9690552
2: -329.7169800, 458.5282593, -212.0506592, 296.0675659, -625.3278809, 670.2380981
3: -463.7893372, 579.4909668, -298.9237671, 375.3389282, -839.1282959, 877.4871826
4: -421.0507202, 622.5073853, -272.8970947, 400.6386414, -821.5962524, 894.4674683

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331576, upper bound: 607.0295115
time: 0.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
time: 0.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.20 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 0, lower bound: -607.0338198, upper bound: 607.0338198
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 0, lower bound: -607.0338198, upper bound: 607.0338198
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 0, lower bound: -607.0331576, upper bound: 607.0295115
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -298.4826050, 350.0044556, -298.4826050, 350.0044556, -648.4870605, 648.4870605
1: -248.2315521, 282.9363708, -248.2315521, 282.9363708, -531.1677856, 531.1677856
2: -199.9671478, 278.1452026, -199.9671478, 278.1452026, -478.1123657, 478.1123657
3: -281.0348511, 353.4127197, -281.0348511, 353.4127197, -634.4472656, 634.4472656
4: -257.1254883, 376.2516174, -257.1254883, 376.2516174, -633.3770752, 633.3770752

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338142, upper bound: 607.0375985
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338166, upper bound: 607.0354187
time: 1.31 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -298.4826050, 350.0044556, -488.1497498, 571.3159790, -869.7985229, 837.1224976
1: -248.2315521, 282.9363708, -412.3843384, 461.1131897, -709.3447266, 692.8405151
2: -199.9671478, 278.1452026, -327.9597778, 455.8415833, -655.4237671, 605.6566162
3: -281.0348511, 353.4127197, -461.1690063, 576.2672729, -856.3715820, 814.5815430
4: -257.1254883, 376.2516174, -418.7281799, 618.8871460, -875.0963135, 794.9244385

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338142, upper bound: 607.0375985
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338166, upper bound: 607.0354185
time: 0.82 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -487.4877319, 570.4686890, -293.5372009, 344.1283875, -830.5938721, 863.9850464
1: -411.8255615, 460.4311523, -244.1956329, 278.1516113, -687.5626831, 704.6267090
2: -327.5029297, 455.1535645, -196.6262970, 273.4741821, -600.5314331, 651.4650269
3: -460.4902344, 575.4501953, -276.3583069, 347.5349731, -808.0252075, 850.8881226
4: -418.1285400, 617.9586792, -252.9428864, 369.9265137, -787.9951782, 869.9917603

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327295, upper bound: 607.0290417
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327172, upper bound: 607.0287650
time: 0.85 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -475.5348511, 558.7756348, -460.6675720, 536.3476562, -1009.5911865, 1016.4235840
1: -402.5134583, 450.9027100, -390.6603394, 433.1323242, -832.0991821, 836.7907715
2: -319.9173889, 446.2551575, -309.6760254, 426.8840942, -745.4365234, 753.9674072
3: -450.6918945, 563.2034302, -433.3648376, 543.2687378, -992.5115967, 995.1846924
4: -408.6188660, 605.8748779, -395.5233765, 578.7026367, -985.7706299, 998.5495605

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0287734, upper bound: 607.0290395
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0287631, upper bound: 607.0287631
time: 0.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.26 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0338142, upper bound: 607.0375985
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0338166, upper bound: 607.0354187
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0338142, upper bound: 607.0375985
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0338166, upper bound: 607.0354185
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0327295, upper bound: 607.0290417
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0327172, upper bound: 607.0287650
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0287734, upper bound: 607.0290395
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -607.0287631, upper bound: 607.0287631

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -288.7386475, 337.6203308, -298.0054932, 349.3982849, -638.1369629, 635.6258545
1: -239.8389587, 272.8605347, -247.8205109, 282.4431458, -522.2820435, 520.6810303
2: -193.2061768, 268.0191345, -199.6358948, 277.6496887, -470.8558655, 467.6550293
3: -270.9801941, 340.9714355, -280.5416870, 352.8038635, -623.7839966, 621.5131226
4: -248.3385620, 362.4671021, -256.6949463, 375.5767212, -623.9152832, 619.1620483

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0354228, upper bound: 607.0354228
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0354228, upper bound: 607.0354231
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -360.2787170, 424.2704773, -297.2629700, 348.4032898, -708.6819458, 721.5334473
1: -303.5158386, 342.2127991, -247.1630402, 281.6486206, -584.6191406, 589.3758545
2: -242.0388641, 337.6308594, -199.1171875, 276.8521729, -518.8909912, 536.7479858
3: -340.7263489, 428.5746765, -279.7632751, 351.8073425, -692.5336304, 708.2269287
4: -310.1425476, 457.8668518, -256.0168152, 374.4954834, -684.6379395, 713.8836060

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0332052, upper bound: 607.0350977
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0351342, upper bound: 607.0351342
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -288.7386475, 337.6203308, -486.9736633, 569.7877197, -858.5263672, 823.5518799
1: -239.8389587, 272.8605347, -411.3756714, 459.8797302, -699.7185669, 681.7449341
2: -193.2061768, 268.0191345, -327.1378479, 454.5945435, -647.4114380, 594.7129517
3: -270.9801941, 340.9714355, -459.9476318, 574.7696533, -844.8153076, 800.9189453
4: -248.3385620, 362.4671021, -417.6501465, 617.1989746, -864.6295166, 780.0771484

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0334020, upper bound: 607.0372628
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331547, upper bound: 607.0372306
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -360.2787170, 424.2704773, -482.4058838, 565.1141357, -924.3631592, 904.8881226
1: -303.5158386, 342.2127991, -407.7542114, 455.9627075, -757.9122925, 747.0928955
2: -242.0388641, 337.6308594, -324.2055664, 450.9191589, -692.1032104, 660.9552002
3: -340.7263489, 428.5746765, -456.0386658, 569.7104492, -909.5328369, 884.2637939
4: -310.1425476, 457.8668518, -413.9132385, 612.1895142, -921.0733032, 870.8678589

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0334048, upper bound: 607.0351177
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331543, upper bound: 607.0351055
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -482.0995483, 564.1041260, -293.2299194, 343.7675476, -824.8211670, 857.3042603
1: -407.3847351, 455.2795715, -243.9411011, 277.8592834, -682.7939453, 699.2207031
2: -323.8693848, 450.0702209, -196.4196625, 273.1860352, -596.6010742, 646.1728516
3: -455.3433228, 569.0806885, -276.0658875, 347.1728210, -802.5161133, 844.2143555
4: -413.4305115, 611.0602417, -252.6764374, 369.5357056, -782.8955078, 862.8255615

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327295, upper bound: 607.0290312
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327295, upper bound: 607.0290417
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -494.9443970, 579.2700195, -287.7671204, 337.1765442, -831.2258911, 867.0371094
1: -418.1730042, 467.5041504, -239.3468323, 272.5324402, -688.3643188, 706.8509521
2: -332.5539246, 462.1858826, -192.7284088, 267.9202271, -600.0595703, 654.6314697
3: -467.6342468, 584.3438110, -270.7172241, 340.5428772, -808.1770020, 854.1857300
4: -424.5537415, 627.5714722, -247.8899384, 362.3861694, -786.9156494, 874.6021118

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327094, upper bound: 607.0287483
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0297564, upper bound: 607.0287479
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -470.2977905, 552.6308594, -460.3439331, 535.9614258, -1003.9474487, 1009.9474487
1: -398.2067261, 445.9300232, -390.3915710, 432.8195496, -827.4539185, 831.5422974
2: -316.3922119, 441.3530273, -309.4568176, 426.5743713, -741.5955811, 748.8439941
3: -445.7220764, 557.0599365, -433.0518799, 542.8823242, -987.1526489, 988.7190552
4: -404.0681152, 599.2258911, -395.2400208, 578.2822876, -980.7896118, 991.6167603

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0287631, upper bound: 607.0287631
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0287631, upper bound: 607.0287631
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -482.6576233, 567.0962524, -455.4203796, 530.0148315, -1010.4947510, 1019.5286865
1: -408.5749817, 457.5975342, -386.2360535, 428.0106506, -833.1014404, 839.0703125
2: -324.7533569, 452.9003906, -306.0995789, 421.8131714, -745.2279053, 757.0488892
3: -457.4738159, 571.6369019, -428.2302551, 536.8853149, -992.8982544, 998.5165405
4: -414.7618103, 614.9660645, -390.8921204, 571.8222656, -985.0546265, 1003.0384521

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0278233, upper bound: 607.0286463
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0287557, upper bound: 607.0287557
time: 0.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.93 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0354228, upper bound: 607.0354228
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0354228, upper bound: 607.0354231
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0332052, upper bound: 607.0350977
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0351342, upper bound: 607.0351342
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0334020, upper bound: 607.0372628
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0331547, upper bound: 607.0372306
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0334048, upper bound: 607.0351177
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0331543, upper bound: 607.0351055
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0327295, upper bound: 607.0290312
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0327295, upper bound: 607.0290417
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0327094, upper bound: 607.0287483
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0297564, upper bound: 607.0287479
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0287631, upper bound: 607.0287631
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0287631, upper bound: 607.0287631
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0278233, upper bound: 607.0286463
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -607.0287557, upper bound: 607.0287557

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -288.7386475, 337.6203308, -288.7386475, 337.6203308, -626.3590088, 626.3590088
1: -239.8389587, 272.8605347, -239.8389587, 272.8605347, -512.6994629, 512.6994629
2: -193.2061768, 268.0191345, -193.2061768, 268.0191345, -461.2253113, 461.2253113
3: -270.9801941, 340.9714355, -270.9801941, 340.9714355, -611.9515991, 611.9515991
4: -248.3385620, 362.4671021, -248.3385620, 362.4671021, -610.8056641, 610.8056641

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0354218, upper bound: 607.0379622
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0354231, upper bound: 607.0365657
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -288.7386475, 337.6203308, -360.2787170, 424.2704773, -713.0091553, 697.8990479
1: -239.8389587, 272.8605347, -303.5158386, 342.2127991, -582.0517578, 575.8283081
2: -193.2061768, 268.0191345, -242.0388641, 337.6308594, -530.8369751, 510.0579834
3: -270.9801941, 340.9714355, -340.7263489, 428.5746765, -699.4393311, 681.6977539
4: -248.3385620, 362.4671021, -310.1425476, 457.8668518, -706.2053833, 672.6095581

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0354218, upper bound: 607.0379622
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0354231, upper bound: 607.0365657
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -355.8263550, 418.9519348, -276.5250549, 323.7129211, -679.5393066, 695.4769287
1: -299.8147583, 337.9142151, -229.9540558, 261.7452087, -561.0126343, 567.8682251
2: -239.0358582, 333.3941040, -185.1757660, 257.2235107, -496.2593079, 518.5696411
3: -336.4087830, 423.2354736, -259.8376465, 327.0856323, -663.4943848, 682.9699097
4: -306.2554932, 452.1163940, -237.9747162, 347.8640442, -654.1195068, 690.0910034

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0278057, upper bound: 607.0304369
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0332036, upper bound: 607.0350964
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330491, upper bound: 607.0338438
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -355.0674438, 418.1022034, -391.4649658, 453.4678345, -808.4892578, 809.5671387
1: -299.1316223, 337.2287598, -325.0589905, 366.5476379, -664.8859253, 662.2762451
2: -238.5351868, 332.6856995, -261.1763000, 359.8958130, -598.4001465, 593.7484741
3: -335.7192993, 422.3755188, -365.2592773, 457.9168396, -793.6361084, 787.5142822
4: -305.6514893, 451.1398621, -334.9873352, 487.2941589, -792.9456177, 786.0162354

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327620, upper bound: 607.0350899
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338392, upper bound: 607.0338392
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -288.4331360, 337.2612000, -481.5845032, 563.4221191, -851.8552246, 817.7795410
1: -239.5862579, 272.5698547, -406.9340515, 454.7272034, -694.3134766, 676.9768677
2: -193.0007629, 267.7328796, -323.5036926, 449.5103149, -642.1196289, 590.7838745
3: -270.6895447, 340.6110229, -454.7998962, 568.3990479, -838.1422119, 795.4108887
4: -248.0733032, 362.0791016, -412.9512939, 610.2994995, -857.4631958, 774.9790039

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0333973, upper bound: 607.0372628
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0334020, upper bound: 607.0363070
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -283.0363770, 330.7318420, -494.4285889, 578.5864868, -861.6227417, 824.2459106
1: -235.0404205, 267.2972717, -417.7220459, 466.9474792, -701.9879150, 682.5928955
2: -189.3520203, 262.5148010, -332.1873169, 461.6242676, -650.6163940, 594.2897949
3: -265.3946228, 334.0410767, -467.0903625, 583.6602173, -848.1650391, 801.1313477
4: -243.3359222, 354.9977112, -424.0733643, 626.8078613, -869.2854614, 779.0671387

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331535, upper bound: 607.0372306
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331547, upper bound: 607.0362908
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -359.9821777, 423.9204712, -476.9711914, 558.7042847, -917.6479492, 899.0836182
1: -303.2709045, 341.9295654, -403.2708740, 450.7728882, -752.4702148, 742.2927246
2: -241.8393860, 337.3514099, -320.5404663, 445.7968750, -686.7797241, 657.0026855
3: -340.4432373, 428.2237854, -450.8515015, 563.2934570, -902.8211670, 878.7230835
4: -309.8847656, 457.4880371, -409.1787720, 605.2385864, -913.8646851, 865.7430420

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0330714
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315685
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -354.6183472, 417.4191589, -489.6364746, 573.5683594, -927.2174072, 905.4124756
1: -298.7481995, 336.6780396, -413.8739929, 462.7543335, -759.9830322, 747.7588501
2: -238.2039032, 332.1503906, -329.0899658, 457.6618958, -695.0454712, 660.3895874
3: -335.1740112, 421.6837463, -462.8970642, 578.2568359, -912.5724487, 884.2435913
4: -305.1768799, 450.4298096, -420.1128540, 621.4148560, -925.3820190, 869.6657715

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0297925, upper bound: 607.0331530
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0297932, upper bound: 607.0315673
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -478.8452454, 559.9230957, -281.4653015, 328.6870422, -806.4945068, 841.3884277
1: -404.6290588, 451.9176025, -233.7952271, 265.6318665, -667.7474976, 685.7128296
2: -321.6230774, 446.6774597, -188.2777252, 260.8881836, -582.0729370, 634.5753174
3: -451.9984131, 565.0479126, -263.8924561, 332.0392761, -784.0376587, 828.0068359
4: -410.4834290, 606.4816895, -241.9462891, 352.8648987, -763.3242188, 847.5384521

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325817, upper bound: 607.0289215
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325875, upper bound: 607.0288868
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -492.6673279, 578.1227417, -486.4070129, 569.7107544, -1059.7343750, 1061.7573242
1: -416.4055481, 466.5628967, -410.7548828, 459.6023865, -872.3123779, 873.4652710
2: -331.1995239, 461.4167786, -326.4793396, 454.1856995, -783.8388062, 786.3674316
3: -466.3403320, 582.7459717, -459.4711304, 574.3203125, -1039.2191162, 1040.9534912
4: -423.1397095, 626.3497314, -417.1314697, 616.4951782, -1037.3773193, 1041.2418213

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325817, upper bound: 607.0289261
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325875, upper bound: 607.0288921
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -492.9634094, 576.7556763, -278.4743652, 325.3635864, -817.4403687, 855.2299805
1: -416.5170593, 465.4834900, -231.3153229, 263.0072021, -677.1098022, 696.7985840
2: -331.1913147, 460.1485596, -186.3110657, 258.2893982, -589.0779419, 646.1087646
3: -465.6402283, 581.9142456, -261.1438599, 328.7384644, -794.3786621, 842.1892090
4: -422.7714233, 624.8244629, -239.4676819, 349.3038635, -772.0753174, 863.4434814

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322090, upper bound: 607.0257603
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322540, upper bound: 607.0284226
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -491.6399536, 576.0323486, -433.8723755, 509.2233276, -998.1265869, 1007.9490356
1: -415.6469421, 464.8019714, -366.6204529, 409.9660339, -822.3027344, 827.7420654
2: -330.5385132, 459.7803650, -290.5219421, 404.6329346, -733.6928711, 748.9617920
3: -465.0359192, 580.6594238, -408.7902832, 514.1815796, -978.1168213, 988.1165161
4: -421.9933472, 624.3286743, -371.6243896, 549.1475220, -969.2727661, 994.0122070

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0293180, upper bound: 607.0257619
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0293636, upper bound: 607.0284242
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -470.2977905, 552.6308594, -455.7991943, 530.6099854, -998.5866089, 1005.3920898
1: -398.2067261, 445.9300232, -386.6332703, 428.4857788, -823.1107178, 827.7666016
2: -316.3922119, 441.3530273, -306.3940735, 422.3021240, -737.3216553, 745.7733154
3: -445.7220764, 557.0599365, -428.7241821, 537.5027466, -981.7614746, 984.3884888
4: -404.0681152, 599.2258911, -391.2819824, 572.4877930, -974.9935303, 987.6460571

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259283, upper bound: 607.0274418
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259257, upper bound: 607.0262945
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -470.2977905, 552.6308594, -470.4238892, 547.6322632, -1015.6412964, 1019.9942017
1: -398.2067261, 445.9300232, -398.7605591, 442.2052917, -836.8305054, 839.8936157
2: -316.3922119, 441.3530273, -316.1748657, 435.8541260, -750.8942261, 755.5220337
3: -445.7220764, 557.0599365, -442.4765625, 554.5836792, -998.8446655, 998.1169434
4: -404.0681152, 599.2258911, -403.7948914, 590.9102173, -993.4287720, 1000.1323242

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259283, upper bound: 607.0274418
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259257, upper bound: 607.0262946
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -482.6576233, 567.0962524, -448.7408752, 521.5830078, -1001.9282837, 1012.6447144
1: -408.5749817, 457.5975342, -380.5626526, 421.0725098, -826.0409546, 833.1179810
2: -324.7533569, 452.9003906, -301.2531738, 414.8150330, -738.1621704, 752.1566772
3: -457.4738159, 571.6369019, -421.3455200, 528.4655762, -984.2916260, 991.6033936
4: -414.7618103, 614.9660645, -384.6256409, 562.2856445, -975.4462891, 996.6965942

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270238, upper bound: 607.0256632
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270694, upper bound: 607.0283223
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -469.7629700, 551.1353760, -540.9244995, 632.9261475, -1098.2144775, 1085.8624268
1: -397.8251343, 444.6590576, -465.0440674, 509.6084900, -902.4566040, 899.7539062
2: -316.0570984, 440.1540527, -363.7484436, 503.4799194, -816.9002075, 800.7808228
3: -444.8309631, 555.5579224, -511.0013123, 641.1465454, -1082.2410889, 1064.3831787
4: -403.3689270, 597.7687378, -462.8971863, 684.2918701, -1084.1279297, 1056.8847656

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0283894, upper bound: 607.0257744
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0284345, upper bound: 607.0284345
time: 2.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.91 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0354218, upper bound: 607.0379622
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0354231, upper bound: 607.0365657
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0354218, upper bound: 607.0379622
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0354231, upper bound: 607.0365657
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0332036, upper bound: 607.0350964
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0330491, upper bound: 607.0338438
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0327620, upper bound: 607.0350899
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0338392, upper bound: 607.0338392
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0333973, upper bound: 607.0372628
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0334020, upper bound: 607.0363070
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0331535, upper bound: 607.0372306
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0331547, upper bound: 607.0362908
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0330714
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315685
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0297925, upper bound: 607.0331530
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0297932, upper bound: 607.0315673
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0325817, upper bound: 607.0289215
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0325875, upper bound: 607.0288868
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0325817, upper bound: 607.0289261
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0325875, upper bound: 607.0288921
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0322090, upper bound: 607.0257603
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0322540, upper bound: 607.0284226
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0293180, upper bound: 607.0257619
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0293636, upper bound: 607.0284242
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0259283, upper bound: 607.0274418
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0259257, upper bound: 607.0262945
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0259283, upper bound: 607.0274418
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0259257, upper bound: 607.0262946
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0270238, upper bound: 607.0256632
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0270694, upper bound: 607.0283223
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0283894, upper bound: 607.0257744
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 0, lower bound: -607.0284345, upper bound: 607.0284345

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -278.7987061, 324.9984436, -285.8948059, 334.0245972, -612.8233032, 610.8932495
1: -231.3677216, 262.6671753, -237.4176178, 269.9519348, -501.3196106, 500.0847473
2: -186.3918304, 257.8292236, -191.2575378, 265.1125183, -451.5043335, 449.0867615
3: -260.8332214, 328.3054199, -268.0838318, 337.3593140, -598.1924438, 596.3892822
4: -239.4595947, 348.6072083, -245.8028412, 358.5129700, -597.9723511, 594.4100342

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0340474, upper bound: 607.0377258
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0361834, upper bound: 607.0377608
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -334.4466248, 392.1169434, -287.8430176, 336.3758850, -670.8225098, 679.9598999
1: -280.4253235, 316.6365356, -239.0321808, 271.8532104, -552.2785645, 555.6687012
2: -224.5718384, 311.7787476, -192.5614014, 267.0095520, -491.5813599, 504.3401489
3: -314.9884949, 396.3434753, -269.9966431, 339.7013245, -654.6898193, 666.3400879
4: -287.9250793, 422.3623352, -247.4835205, 361.0965576, -649.0216064, 669.8458252

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0340473, upper bound: 607.0361447
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0361834, upper bound: 607.0361834
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -278.7987061, 324.9984436, -358.2850952, 421.6980286, -700.4967041, 683.2835693
1: -231.3677216, 262.6671753, -301.7912292, 340.1177063, -571.4854126, 563.8927612
2: -186.3918304, 257.8292236, -240.6434021, 335.5158386, -521.9076538, 498.4725952
3: -260.8332214, 328.3054199, -338.6114502, 425.9809265, -686.6829834, 666.9168701
4: -239.4595947, 348.6072083, -308.3221741, 454.9848328, -694.4443359, 656.9293823

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338988, upper bound: 607.0340562
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0318440, upper bound: 607.0340673
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -334.4466248, 392.1169434, -351.7103882, 413.8967285, -748.3433838, 743.8253784
1: -280.4253235, 316.6365356, -296.4736633, 333.9097900, -614.1258545, 612.2781372
2: -224.5718384, 311.7787476, -236.3335419, 329.4589539, -553.8812866, 547.9492798
3: -314.9884949, 396.3434753, -332.5498047, 418.2621155, -733.1295166, 728.8932495
4: -287.9250793, 422.3623352, -302.7865601, 446.7689209, -734.6939697, 725.1488647

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0339059, upper bound: 607.0329466
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0318475, upper bound: 607.0329445
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -313.0316162, 369.3862915, -266.7738647, 312.0428772, -625.0744629, 636.1600952
1: -264.6540833, 297.8183289, -221.7143250, 252.3688965, -516.2539673, 519.5325317
2: -210.4920349, 294.1796570, -178.6143494, 247.9226685, -458.4147034, 472.7940063
3: -296.6228027, 373.2322998, -250.4773712, 315.3556519, -611.9784546, 623.5304565
4: -269.5099487, 399.2759094, -229.5261230, 335.2301636, -604.7399902, 628.8020020

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0324816, upper bound: 607.0346132
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317347, upper bound: 607.0345902
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -353.1690063, 415.6126099, -275.6845398, 322.6776123, -675.8466187, 691.2971191
1: -297.5197754, 335.2400513, -229.2143860, 260.9111938, -557.8661499, 564.4544678
2: -237.2017822, 330.7523804, -184.5956268, 256.3998108, -493.6015930, 515.3479614
3: -333.7276611, 419.8548584, -259.0002136, 326.0301208, -659.7578125, 678.7234497
4: -303.8489075, 448.5238647, -237.2201996, 346.7432556, -650.5921631, 685.7440796

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325018, upper bound: 607.0335024
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317241, upper bound: 607.0334833
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -311.6414185, 367.6971130, -376.5374756, 435.8553467, -747.3846436, 744.2346191
1: -263.4636536, 296.4594116, -312.4828796, 352.3768005, -614.8107910, 608.8075562
2: -209.5480194, 292.8077087, -251.1555634, 345.9168701, -555.3826294, 543.8098755
3: -295.2541809, 371.5484619, -351.1771851, 440.1386414, -735.3927612, 722.5679321
4: -268.3142395, 397.3937683, -322.0949707, 468.3416748, -736.6558228, 719.3529663

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330230, upper bound: 607.0346159
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0335103, upper bound: 607.0346159
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -351.6210632, 413.8181763, -390.8877258, 452.7313232, -804.2828369, 804.7059326
1: -296.1706238, 333.7862854, -324.5666809, 365.9584351, -661.3314209, 658.3071289
2: -236.1649780, 329.2835083, -260.7772217, 359.3158875, -595.4322510, 589.9441528
3: -332.2768555, 418.0438843, -364.6719360, 457.1712341, -789.4478760, 782.5733032
4: -302.5554199, 446.5174561, -334.4570923, 486.5075989, -789.0629883, 780.8399048

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330114, upper bound: 607.0335027
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0335010, upper bound: 607.0335010
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -278.4971008, 324.6439209, -479.2256775, 560.4054565, -838.8874512, 802.7463379
1: -231.1184387, 262.3804016, -404.9141235, 452.2746887, -683.3931274, 664.7264404
2: -186.1892090, 257.5466919, -321.8509521, 447.0399170, -632.8206787, 578.9301758
3: -260.5464783, 327.9496155, -452.3731384, 565.3773804, -824.9412842, 780.3227539
4: -239.1978302, 348.2243958, -410.8117371, 606.9390259, -845.2194824, 758.9771118

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0333205, upper bound: 607.0371528
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331696, upper bound: 607.0371513
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -334.1460571, 391.7627563, -477.1678162, 558.3300171, -891.8985596, 867.5463867
1: -280.1765442, 316.3497314, -403.2707520, 450.6310120, -729.9572754, 716.8063965
2: -224.3695984, 311.4960327, -320.5808411, 445.5295105, -669.2153931, 631.3480225
3: -314.7017517, 395.9877014, -450.6505432, 563.1024780, -876.8261108, 846.5426636
4: -287.6639709, 421.9792175, -409.1807251, 604.8616333, -891.4475098, 830.5454712

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0333223, upper bound: 607.0362478
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331704, upper bound: 607.0362489
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -273.2373047, 318.2947693, -492.1231384, 575.6354980, -848.8727417, 809.4492188
1: -226.6941223, 257.2575989, -415.7481079, 464.5462341, -691.2402954, 670.5396118
2: -182.6394196, 252.4826355, -330.5710754, 459.2064819, -641.4696655, 582.6274414
3: -255.4055939, 321.5610046, -464.7187500, 580.7019653, -835.1832886, 786.2797852
4: -234.5881805, 341.3553772, -421.9806213, 623.5202637, -857.2432861, 763.3252563

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299192, upper bound: 607.0366215
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327368, upper bound: 607.0366777
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -328.8328552, 385.3202209, -490.0521240, 573.5092163, -901.8271484, 874.1371460
1: -275.6816101, 311.1402588, -414.0738831, 462.8713074, -737.7588501, 722.5039062
2: -220.7651825, 306.3359070, -329.2872314, 457.6562500, -677.7656250, 634.9340210
3: -309.4653931, 389.4993286, -462.9627380, 578.3857422, -886.9332886, 852.3810425
4: -282.9931946, 414.9760437, -420.3360901, 621.3826294, -903.3422241, 834.7440186

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299260, upper bound: 607.0356791
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327439, upper bound: 607.0357280
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -344.5086060, 405.5797424, -471.4977112, 552.0193481, -895.3875732, 875.1497803
1: -290.4953613, 327.1592407, -398.6602783, 445.3744507, -734.2607422, 722.8746948
2: -231.4824219, 322.7980652, -316.8118896, 440.4336548, -671.0426636, 638.6739502
3: -325.6855774, 409.8242798, -445.4419250, 556.5625610, -881.2650757, 854.8868408
4: -296.4940491, 437.8014526, -404.3358459, 597.9685669, -893.1897583, 841.1677856

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0255099, upper bound: 607.0292797
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0314377
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0314381
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -455.8928528, 530.4455566, -468.6424561, 548.8765259, -1002.3151855, 996.0791016
1: -379.1902771, 428.5035095, -396.3843689, 442.8457642, -820.1475830, 821.1021118
2: -305.0424805, 420.5095520, -314.9370728, 437.9778137, -741.4110718, 734.1629028
3: -424.7537231, 536.0644531, -442.8608704, 553.4400024, -976.8629150, 977.5885620
4: -390.8707581, 568.3905029, -401.9461670, 594.6123047, -983.0646362, 969.3107910

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315684
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315685
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -339.2901917, 399.3001404, -484.2173462, 566.9903564, -905.2069092, 881.7391357
1: -286.0970459, 322.0851440, -409.3158569, 457.4410095, -741.9813843, 728.5720215
2: -227.9493103, 317.7813721, -325.4043579, 452.3956604, -679.5053101, 642.2863159
3: -320.6151428, 403.4954529, -457.6073608, 571.6126709, -891.3023071, 860.7391357
4: -291.9307861, 430.9981995, -415.3415833, 614.2817993, -904.9926147, 845.4119263

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0253706, upper bound: 607.0299679
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263010, upper bound: 607.0320717
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0260237, upper bound: 607.0316756
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -451.4359436, 524.9840698, -481.6734619, 564.0874634, -1013.1210938, 1003.7980957
1: -375.3985596, 424.0933533, -407.2902222, 455.0915222, -828.6508789, 827.7085571
2: -301.9986572, 416.1188049, -323.7208862, 450.0879517, -750.5059204, 738.5868530
3: -420.3444824, 530.5739136, -455.1521606, 568.7568359, -987.8201294, 984.4012451
4: -386.9244995, 562.4219360, -413.1566467, 611.1192017, -995.6589355, 974.5971069

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0262549, upper bound: 607.0282787
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259931, upper bound: 607.0282511
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -436.1860962, 510.2218323, -271.8155823, 317.1621704, -751.9569702, 781.8277588
1: -369.9296570, 411.6639404, -225.6569214, 256.3847351, -623.4194336, 637.2402954
2: -293.1571045, 407.3157959, -181.8043060, 251.7207031, -544.2965088, 588.6621094
3: -412.3002930, 514.9616089, -254.6623077, 320.4636841, -732.7639771, 768.4679565
4: -373.7900391, 553.4592285, -233.6138153, 340.4208374, -713.9703979, 786.0450439

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0357031, upper bound: 607.0289111
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0357111, upper bound: 607.0289114
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0342308, upper bound: 607.0289130
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -475.8775330, 556.3068848, -280.6028137, 327.6242981, -802.3036499, 836.8342285
1: -402.0405273, 448.9913940, -233.0377350, 264.7763062, -664.3026733, 682.0288696
2: -319.5777283, 443.7769165, -187.6838226, 260.0412292, -579.1090088, 631.0493164
3: -449.0682983, 561.3911743, -263.0330505, 330.9586182, -780.0267334, 823.3946533
4: -407.8392944, 602.5403442, -241.1748047, 351.7118225, -759.3851318, 842.7508545

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0357010, upper bound: 607.0288771
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0357088, upper bound: 607.0288764
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0342366, upper bound: 607.0288785
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -450.6463318, 529.5029297, -478.4382935, 560.1494751, -1007.7102051, 1004.8261719
1: -382.2748108, 427.2038269, -404.1209412, 451.9055786, -830.0524902, 827.1411743
2: -303.1940613, 422.9325256, -321.1049805, 446.5850830, -748.0680542, 742.3636475
3: -427.4125671, 533.7854614, -451.8404541, 564.7251587, -990.4992065, 984.1101074
4: -387.0945435, 574.5115356, -410.1835938, 606.2086792, -990.8084717, 982.2302856

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325796, upper bound: 607.0289261
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325796, upper bound: 607.0289261
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -490.1675720, 575.1331787, -485.5586853, 568.7133789, -1056.0472412, 1057.7734375
1: -414.2161255, 464.1471558, -410.0112915, 458.7956238, -869.3004761, 870.2320557
2: -329.4793396, 459.0267639, -325.8988953, 453.3882446, -781.2400513, 783.3466187
3: -463.9006653, 579.7044067, -458.6550903, 573.3001709, -1035.6462402, 1036.9942627
4: -420.9264832, 623.0946045, -416.3881836, 615.4088135, -1033.9238281, 1037.1445312

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325857, upper bound: 607.0288921
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325857, upper bound: 607.0288921
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -471.1581421, 550.7297363, -274.3638611, 320.4646912, -790.6739502, 825.0936279
1: -398.3134155, 444.4950867, -227.9095612, 259.0555420, -654.9255981, 672.4046631
2: -316.4833069, 439.4104919, -183.5428314, 254.3998413, -570.4618530, 622.5949097
3: -444.4937744, 555.8161621, -257.1878052, 323.8320007, -768.3258057, 812.1035767
4: -403.7216492, 596.6267700, -235.8825378, 344.0259705, -747.7247925, 831.6721191

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322090, upper bound: 607.0257514
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322088, upper bound: 607.0257603
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -579.4462891, 672.8665771, -273.6601868, 319.6546631, -898.5459595, 946.5267334
1: -487.3479309, 543.3624878, -227.2558441, 258.4039001, -743.1503296, 770.6182861
2: -388.0650635, 536.1054077, -183.0709229, 253.7087708, -641.4729004, 718.8098755
3: -543.8536987, 679.2064209, -256.5037537, 323.0032349, -866.8569336, 934.9517212
4: -495.2201843, 727.8547363, -235.3127594, 343.0683594, -838.2885132, 962.2365112

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322540, upper bound: 607.0284133
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322540, upper bound: 607.0284226
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -469.5534668, 549.5781250, -429.5548706, 504.0572510, -970.7955933, 977.1403198
1: -397.1881714, 443.4842834, -363.0400085, 405.7814941, -799.6195679, 802.7974854
2: -315.6200256, 438.6871643, -287.5984192, 400.5033569, -714.6184692, 724.9338379
3: -443.5700378, 554.1387329, -404.5986938, 508.9984436, -951.4722290, 957.3630981
4: -402.6692810, 595.6556396, -367.8362732, 543.5479126, -944.3110352, 961.5432739

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0249715, upper bound: 607.0215685
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0293152, upper bound: 607.0257503
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0282879, upper bound: 607.0257393
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -578.4691772, 672.5370483, -429.1814575, 503.6302795, -1079.6682129, 1099.7679443
1: -486.8137512, 543.0232544, -362.6715393, 405.4780273, -888.7283936, 902.1077271
2: -387.7046814, 536.0282593, -287.3613586, 400.1685486, -786.4794922, 822.0256958
3: -543.5397339, 678.2865601, -404.2920227, 508.5834656, -1051.1302490, 1081.3803711
4: -494.7590942, 727.7448730, -367.5717773, 543.0971680, -1036.3391113, 1093.2805176

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0293324, upper bound: 607.0273914
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0283049, upper bound: 607.0273714
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -457.3473816, 536.7317505, -449.3703308, 523.0917969, -977.8958740, 982.8916626
1: -387.2347412, 433.1159973, -381.3495483, 422.4331665, -806.0064697, 809.5398560
2: -307.5486450, 428.6098022, -302.0808105, 416.3377991, -722.4212646, 728.6622314
3: -432.9116211, 541.0309448, -422.6464233, 529.9592896, -961.3381348, 962.1658936
4: -392.5912781, 581.9659424, -385.7208557, 564.4331665, -955.3176880, 964.7802734

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265345, upper bound: 607.0285815
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267388, upper bound: 607.0286335
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -566.9891357, 659.1947632, -447.6782837, 521.0408936, -1084.4260254, 1102.8334961
1: -474.5554504, 532.5085449, -380.0029907, 420.7657471, -891.5458374, 906.9367676
2: -379.9182129, 524.4686890, -300.9686279, 414.6893616, -792.5765381, 823.1603394
3: -529.6099243, 664.7123413, -420.9683533, 527.9096680, -1055.7864990, 1083.4671631
4: -485.2721252, 710.0242920, -384.2620544, 562.1430664, -1044.8300781, 1091.3566895

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265057, upper bound: 607.0266449
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267096, upper bound: 607.0267096
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -457.3473816, 536.7317505, -464.3143616, 540.5010986, -995.3410034, 997.8076172
1: -387.2347412, 433.1159973, -393.7580872, 436.4730530, -820.0527954, 821.9512939
2: -307.5486450, 428.6098022, -312.0929260, 430.2253418, -736.3300171, 738.6375122
3: -432.9116211, 541.0309448, -436.7600098, 547.4291992, -978.8120728, 976.2525635
4: -392.5912781, 581.9659424, -398.5385742, 583.3197021, -974.2200928, 977.5692139

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0238138, upper bound: 607.0272766
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0260092, upper bound: 607.0274265
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -566.9891357, 659.1947632, -462.7351685, 538.5324707, -1101.9362793, 1117.8662109
1: -474.5554504, 532.5085449, -392.4679565, 434.8345947, -905.6222534, 919.4035034
2: -379.9182129, 524.4686890, -311.0266113, 428.5918274, -806.4974365, 833.1853027
3: -529.6099243, 664.7123413, -435.0776367, 545.4487915, -1073.3254395, 1097.5527344
4: -485.2721252, 710.0242920, -397.1268005, 581.0507202, -1063.7354736, 1104.1932373

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0238089, upper bound: 607.0261235
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0260046, upper bound: 607.0262781
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -459.6546936, 539.8221436, -444.7634888, 516.7609863, -974.0578003, 981.3513794
1: -389.4608765, 435.5898743, -377.2483521, 417.1818237, -802.9990845, 807.7649536
2: -309.2645264, 431.1953125, -298.5466309, 410.9701233, -718.8275146, 727.7269897
3: -435.3507690, 544.2793579, -417.4517212, 523.6382446, -957.3283081, 960.3360596
4: -394.7182922, 585.4822998, -381.1082458, 557.0635376, -950.1756592, 963.6852417

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0220257, upper bound: 607.0214745
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270238, upper bound: 607.0256632
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270238, upper bound: 607.0256632
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -570.7013550, 664.6488037, -444.2019958, 516.2385254, -1084.7864990, 1105.5993652
1: -480.5810852, 536.6408081, -376.7290039, 416.7630615, -893.4779053, 908.3332520
2: -382.6597595, 529.9539185, -298.1646423, 410.5314026, -791.8114624, 826.0722656
3: -536.8654785, 670.3054199, -416.9691772, 523.1005859, -1058.3741455, 1085.9616699
4: -488.4362793, 719.4475708, -380.6703186, 556.4711304, -1043.5350342, 1097.0933838

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270694, upper bound: 607.0283223
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270694, upper bound: 607.0283224
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -446.8470764, 523.9586792, -536.9851685, 628.1782837, -1070.4879150, 1054.6878662
1: -378.7867737, 422.7587891, -461.7746277, 505.7679443, -879.5310059, 874.5280762
2: -300.6317139, 418.5319824, -361.0815125, 499.6828918, -797.6688843, 776.4550171
3: -422.7934570, 528.3276978, -507.1692200, 636.3839722, -1055.4208984, 1033.2869873
4: -383.4222107, 568.3972778, -459.4180908, 679.1515503, -1059.0205078, 1024.0178223

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0241249, upper bound: 607.0215823
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0283894, upper bound: 607.0257744
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0283894, upper bound: 607.0257744
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -557.9711914, 649.2020264, -536.0529175, 627.0729370, -1180.7375488, 1178.9835205
1: -470.1423035, 524.1572266, -460.9231262, 504.9215088, -969.7649536, 975.1369629
2: -374.1354980, 517.6408691, -360.4546814, 498.8081360, -870.3305664, 874.9083862
3: -524.6306152, 654.7626343, -506.3014221, 635.3098145, -1156.2427979, 1158.9689941
4: -477.3326721, 702.8538208, -458.6436157, 677.9566040, -1151.9801025, 1157.5937500

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0284345, upper bound: 607.0284345
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0284345, upper bound: 607.0284345
time: 0.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.65 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0340474, upper bound: 607.0377258
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0361834, upper bound: 607.0377608
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0340473, upper bound: 607.0361447
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0361834, upper bound: 607.0361834
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0338988, upper bound: 607.0340562
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0318440, upper bound: 607.0340673
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0339059, upper bound: 607.0329466
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0318475, upper bound: 607.0329445
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0324816, upper bound: 607.0346132
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0317347, upper bound: 607.0345902
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0325018, upper bound: 607.0335024
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0317241, upper bound: 607.0334833
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0330230, upper bound: 607.0346159
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0335103, upper bound: 607.0346159
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0330114, upper bound: 607.0335027
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0335010, upper bound: 607.0335010
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0333205, upper bound: 607.0371528
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0331696, upper bound: 607.0371513
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0333223, upper bound: 607.0362478
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0331704, upper bound: 607.0362489
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0299192, upper bound: 607.0366215
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0327368, upper bound: 607.0366777
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0299260, upper bound: 607.0356791
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0327439, upper bound: 607.0357280
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0314377
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0314381
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315684
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315685
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0263010, upper bound: 607.0320717
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0260237, upper bound: 607.0316756
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0262549, upper bound: 607.0282787
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0259931, upper bound: 607.0282511
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0357111, upper bound: 607.0289114
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0342308, upper bound: 607.0289130
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0357088, upper bound: 607.0288764
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0342366, upper bound: 607.0288785
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0325796, upper bound: 607.0289261
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0325796, upper bound: 607.0289261
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0325857, upper bound: 607.0288921
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0325857, upper bound: 607.0288921
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0322090, upper bound: 607.0257514
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0322088, upper bound: 607.0257603
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0322540, upper bound: 607.0284133
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0322540, upper bound: 607.0284226
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0293152, upper bound: 607.0257503
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0282879, upper bound: 607.0257393
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0293324, upper bound: 607.0273914
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0283049, upper bound: 607.0273714
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0265345, upper bound: 607.0285815
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0267388, upper bound: 607.0286335
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0265057, upper bound: 607.0266449
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0267096, upper bound: 607.0267096
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0238138, upper bound: 607.0272766
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0260092, upper bound: 607.0274265
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0238089, upper bound: 607.0261235
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0260046, upper bound: 607.0262781
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0270238, upper bound: 607.0256632
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0270238, upper bound: 607.0256632
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0270694, upper bound: 607.0283223
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0270694, upper bound: 607.0283224
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0283894, upper bound: 607.0257744
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0283894, upper bound: 607.0257744
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0284345, upper bound: 607.0284345
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -607.0284345, upper bound: 607.0284345

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -274.6720886, 320.0463562, -265.3163757, 309.4672241, -584.1392822, 585.3627319
1: -227.9376373, 258.6766968, -220.3445587, 250.1575012, -478.0951538, 479.0212402
2: -183.6074829, 253.8932800, -177.4165497, 245.5918274, -429.1993103, 431.3098145
3: -256.8358459, 323.3510132, -248.2787781, 312.7730713, -569.6088867, 571.6296997
4: -235.8500519, 343.2686157, -227.8797913, 332.0309143, -567.8808594, 571.1483154

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0334450, upper bound: 607.0369825
time: 1.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0336221, upper bound: 607.0367313
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -273.5534363, 318.7601624, -380.4393616, 439.4642029, -713.0176392, 699.1995239
1: -226.9412537, 257.6414490, -315.6032715, 355.1346741, -582.0759277, 573.2447510
2: -182.8580475, 252.8249817, -253.5179901, 348.4447632, -531.3027954, 506.3429565
3: -255.7555847, 322.0449524, -353.8830872, 443.8474426, -699.6030273, 675.9280396
4: -234.9255524, 341.7915955, -325.0481262, 471.7064209, -706.6319580, 666.8395996

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0353486, upper bound: 607.0370186
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0353919, upper bound: 607.0367589
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -330.3197632, 387.1475525, -267.3087769, 311.8952942, -642.2149658, 654.4562988
1: -276.9918213, 312.6274109, -222.0057068, 252.1188354, -529.1106567, 534.6329956
2: -221.7837067, 307.8241577, -178.7556152, 247.5514069, -469.3350525, 486.5797424
3: -310.9696350, 391.3662109, -250.2505951, 315.1925964, -626.1621704, 641.6166992
4: -284.3091736, 416.9980164, -229.6069336, 334.6998596, -619.0090332, 646.6049194

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338373, upper bound: 607.0338373
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0338373, upper bound: 607.0361447
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -329.5758667, 386.2978821, -382.4096985, 441.8994446, -771.4752808, 768.7075806
1: -276.3215027, 311.9317322, -317.2384949, 357.1003418, -633.3417969, 629.1702271
2: -221.2837524, 307.1032410, -254.8392792, 350.4103699, -571.6939697, 561.9425049
3: -310.2437744, 390.4902344, -355.8582764, 446.2559204, -756.4996948, 746.3485107
4: -283.6900940, 416.0109558, -326.7325134, 474.3842773, -758.0743408, 742.7434692

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0357482, upper bound: 607.0359703
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327775, upper bound: 607.0357354
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -272.5751343, 317.5905457, -342.8202209, 403.3825378, -675.9573975, 660.4107666
1: -226.1667023, 256.7067566, -289.0281677, 325.3669739, -551.5336304, 545.1674805
2: -182.2028656, 251.9297485, -230.2945709, 320.9841919, -503.1870728, 482.2243042
3: -254.8177948, 320.8589478, -323.8764954, 407.6061401, -662.2703857, 644.7353516
4: -234.0440521, 340.5987854, -294.9441528, 435.3304443, -669.3745117, 635.5429688

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0304058, upper bound: 607.0298767
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322156, upper bound: 607.0329984
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0322156, upper bound: 607.0335172
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -271.0344849, 315.8416443, -454.2575989, 528.2702026, -798.7433472, 768.9592285
1: -224.9749298, 255.2694702, -377.7571106, 426.7385254, -651.6818848, 632.1530762
2: -181.1938019, 250.5442352, -303.8871460, 418.7256775, -599.6520386, 553.7677002
3: -253.3822632, 319.1145020, -422.9993591, 533.8764038, -786.1453857, 742.1137695
4: -232.7367096, 338.7135010, -389.3602905, 565.9591064, -798.6958008, 727.5950317

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0312135, upper bound: 607.0330291
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0312135, upper bound: 607.0335230
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -328.4659119, 384.9793091, -336.1145020, 395.3810120, -723.8469238, 721.0209961
1: -275.4585876, 310.8826599, -283.5941772, 319.0142517, -594.2517700, 593.6376343
2: -220.5505676, 306.0939636, -225.8928986, 314.7731018, -535.1559448, 531.8153687
3: -309.2028809, 389.1748657, -317.6578674, 399.7120361, -708.7740479, 706.8325806
4: -282.7185364, 414.6616821, -289.2898865, 426.9011230, -709.6196289, 703.9514160

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0304289, upper bound: 607.0288079
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330779, upper bound: 607.0325116
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331562, upper bound: 607.0328023
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -326.7556763, 383.0229492, -448.3531799, 521.2144775, -846.8352051, 829.9636841
1: -274.0953064, 309.2845154, -372.8963318, 421.1071167, -694.1348267, 681.0119629
2: -219.4206696, 304.5382690, -299.9670410, 413.1599426, -632.0291138, 603.5791016
3: -307.5684509, 387.2157288, -417.4397278, 526.8613281, -833.3087158, 804.4010010
4: -281.2603149, 412.5368042, -384.3141174, 558.3954468, -839.6383057, 795.8438721

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0314439, upper bound: 607.0325178
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0315758, upper bound: 607.0327993
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -312.7620544, 369.0655212, -261.7840881, 306.1684875, -618.9305420, 630.8496094
1: -264.4299011, 297.5587158, -217.5763702, 247.6215668, -511.2777405, 515.1350708
2: -210.3102875, 293.9229431, -175.2616882, 243.2391968, -453.5494385, 469.1846313
3: -296.3630371, 372.9104919, -245.7235107, 309.4629517, -605.8256836, 618.4534912
4: -269.2749634, 398.9275208, -225.1992493, 328.8803711, -598.1553345, 624.1267700

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317347, upper bound: 607.0345902
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317347, upper bound: 607.0345902
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -307.6646423, 362.9085999, -272.0942688, 318.4167480, -626.0813599, 635.0027466
1: -260.1459351, 292.5924072, -226.3908234, 257.4488220, -516.7832031, 518.9832153
2: -206.8597717, 289.0009155, -182.2636414, 253.0008850, -459.8606567, 471.2645569
3: -291.3705750, 366.7287598, -255.4645081, 321.7901306, -613.1607056, 621.9749756
4: -264.8080750, 392.2481689, -234.0717316, 342.1911926, -606.9992676, 626.3198853

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317347, upper bound: 607.0345902
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317347, upper bound: 607.0345902
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -352.8713989, 415.2611694, -270.6820374, 316.7896423, -669.6608887, 685.9431152
1: -297.2737427, 334.9556885, -225.0734863, 256.1513672, -552.8557739, 560.0291138
2: -237.0015564, 330.4718628, -181.2365265, 251.7065430, -488.7080078, 511.7083740
3: -333.4433594, 419.5024109, -254.2338104, 320.1273499, -653.5706787, 673.6046753
4: -303.5901184, 448.1435547, -232.8845062, 340.3819275, -643.9719238, 681.0280762

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317241, upper bound: 607.0329965
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317241, upper bound: 607.0334833
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -347.5122375, 408.7347107, -280.8345642, 328.9174805, -676.4296875, 689.5692749
1: -292.7438354, 329.6856384, -233.7922821, 265.8641052, -557.9989624, 563.4779053
2: -233.3614044, 325.2448425, -188.1415100, 261.3884888, -494.7498779, 513.3863525
3: -328.1463623, 412.9424744, -263.8592529, 332.3045349, -660.4509277, 676.6409302
4: -298.8750305, 441.0444031, -241.6291199, 353.5627747, -652.4378052, 682.6734619

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317070, upper bound: 607.0317149
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317070, upper bound: 607.0334833
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -311.3464050, 367.3453979, -372.0061646, 430.4499512, -741.6698608, 739.3515625
1: -263.2193909, 296.1751099, -308.7257080, 347.9934692, -610.1727905, 604.7495728
2: -209.3492737, 292.5266724, -248.0952759, 341.5847168, -550.8479004, 540.4612427
3: -294.9697571, 371.1968689, -346.7874451, 434.7099915, -729.6796265, 717.8244019
4: -268.0566711, 397.0125122, -318.1350098, 462.4640198, -730.5206909, 714.9967041

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330230, upper bound: 607.0346159
time: 5.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330230, upper bound: 607.0346159
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -306.9165649, 362.0301208, -386.4617004, 446.8335571, -753.5523682, 748.4374390
1: -259.5144653, 291.8898010, -320.5425720, 361.2660217, -619.6763916, 612.3145142
2: -206.3614349, 288.2926331, -257.6992188, 354.5761719, -560.8277588, 545.7695923
3: -290.6836243, 365.8571167, -359.8513489, 451.2642517, -741.9478149, 725.4881592
4: -264.1921387, 391.2727966, -330.3112183, 480.0691223, -744.2612305, 721.3531494

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0335102, upper bound: 607.0346159
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0335102, upper bound: 607.0346159
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -351.3146973, 413.4537659, -386.3312378, 447.3085022, -798.5399780, 799.7849731
1: -295.9160461, 333.4915161, -320.7912598, 361.5582886, -656.6692505, 654.2220459
2: -235.9581146, 328.9919739, -257.6986084, 354.9701233, -590.8760376, 586.5702515
3: -331.9817200, 417.6784668, -360.2641296, 451.7280579, -783.7097778, 777.7988281
4: -302.2878418, 446.1217957, -330.4783325, 480.6108704, -782.8985596, 776.4556885

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330114, upper bound: 607.0330114
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330114, upper bound: 607.0335010
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -346.1107178, 407.1385803, -399.8360291, 462.5762634, -808.5330200, 806.9746094
1: -291.5293274, 328.3883667, -331.8100281, 373.9447632, -664.5990601, 660.1689453
2: -232.4291077, 323.9361877, -266.6717834, 367.0750427, -599.4314575, 590.4288330
3: -326.8573303, 411.3298950, -372.4414368, 467.1691895, -794.0264893, 783.5794678
4: -297.7176819, 439.2604980, -341.8434753, 496.9993896, -794.7170410, 780.8714600

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0335010, upper bound: 607.0330114
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0317241, upper bound: 607.0335010
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -268.6606750, 312.9185791, -436.6576843, 510.9434509, -779.3604126, 748.0947266
1: -222.8193359, 252.9704895, -370.3176880, 412.2055969, -634.9248047, 620.3327026
2: -179.5887604, 248.2178650, -293.4546814, 407.8501892, -586.9541626, 541.0576172
3: -251.1502228, 316.1744690, -412.7901001, 515.5573730, -765.5007324, 728.9645996
4: -230.7055054, 335.5630798, -374.2389221, 554.1374512, -783.7829590, 709.5248413

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0329102, upper bound: 607.0335406
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330177, upper bound: 607.0354852
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -277.5943298, 323.5305176, -476.2603455, 556.8017578, -834.2814941, 798.4981689
1: -230.3316498, 261.4846191, -402.3252869, 449.3568726, -679.6885376, 661.2385864
2: -185.5672455, 256.6588440, -319.8078918, 444.1499939, -629.2741699, 575.9253540
3: -259.6449585, 326.8188782, -449.4477234, 561.7238159, -820.2875977, 776.2666016
4: -238.3899536, 347.0160828, -408.1716003, 603.0098877, -840.4038696, 754.9841309

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0311856, upper bound: 607.0335235
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0312131, upper bound: 607.0354654
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -324.6914062, 380.4538574, -435.3913269, 509.8978577, -833.7781372, 814.0791016
1: -272.1784058, 307.2656555, -369.3363342, 411.3933716, -682.4438477, 673.4022827
2: -218.0221252, 302.5005493, -292.7286682, 407.1613159, -624.4199829, 594.3468018
3: -305.6600037, 384.6205444, -411.8979492, 514.3048096, -818.7681885, 796.2905273
4: -279.4754639, 409.7850647, -373.3266602, 553.1855469, -831.4387207, 782.2800293

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0329246, upper bound: 607.0327665
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0330286, upper bound: 607.0345621
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -333.2109985, 390.6163330, -474.3884277, 554.9652100, -887.5076904, 863.4577637
1: -279.3606873, 315.4277954, -400.8436584, 447.9056396, -726.3500977, 713.4577026
2: -223.7260742, 310.5851135, -318.6659851, 442.8316040, -665.8438721, 628.4514771
3: -313.7733154, 394.8230286, -447.9195251, 559.6837769, -872.3872070, 842.5615845
4: -286.8263245, 420.7406921, -406.7092896, 601.1935425, -886.8695679, 826.6951294

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0312001, upper bound: 607.0327539
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0312241, upper bound: 607.0345486
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -269.1272278, 313.3586121, -470.3232117, 549.6004028, -818.7276611, 782.6467896
1: -223.2767334, 253.2791443, -397.5451965, 443.5541382, -666.8308716, 648.3258667
2: -179.8639526, 248.5600128, -315.8601074, 438.4593506, -617.9379883, 563.9744263
3: -251.4177704, 316.6198730, -443.5775757, 554.5899048, -805.0492554, 760.1974487
4: -230.9898529, 336.0329895, -402.9228210, 595.3093872, -825.4428711, 738.9115601

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298777, upper bound: 607.0334992
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298777, upper bound: 607.0366215
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -268.4227295, 312.5999756, -578.2485962, 671.2665405, -939.6892700, 890.2373047
1: -222.6367340, 252.6668243, -486.2676086, 542.0468140, -764.6835327, 736.2874146
2: -179.3976746, 247.9195404, -387.2211609, 534.7868042, -713.7942505, 634.8175659
3: -250.7787476, 315.8372192, -542.5598145, 677.5165405, -927.4859619, 858.3970337
4: -230.4358063, 335.1428223, -494.1185913, 726.0318604, -955.5168457, 829.2614136

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0316801, upper bound: 607.0366444
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0316557, upper bound: 607.0354581
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -324.7225037, 380.3549194, -468.0145264, 547.1281128, -871.3181763, 847.0783081
1: -272.2569885, 307.1360168, -395.6429443, 441.6112061, -713.0544434, 700.0424805
2: -217.9851227, 302.3820496, -314.4019470, 436.6306458, -653.9504395, 616.0768433
3: -305.4462585, 384.5287476, -441.5597839, 551.9368286, -856.4347534, 826.0258179
4: -279.3864136, 409.6096191, -401.0499573, 592.7857056, -871.1492920, 810.0586548

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298932, upper bound: 607.0327344
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298932, upper bound: 607.0356789
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -324.4422607, 380.1181335, -577.4291382, 670.6666260, -994.6707153, 956.6634521
1: -271.9885254, 306.9308777, -485.6929016, 541.6187744, -812.9313354, 789.7062988
2: -217.8054810, 302.1627197, -386.8115540, 534.4748535, -751.6069946, 688.3887329
3: -305.2389526, 384.2524109, -542.0675659, 676.7485352, -981.1959839, 926.3199463
4: -279.1897888, 409.3123169, -493.6031494, 725.5924683, -1003.6456299, 902.7015991

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0316846, upper bound: 607.0357001
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0316665, upper bound: 607.0345437
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -344.5086060, 405.5797424, -463.7898560, 542.6969604, -885.9662476, 867.3240356
1: -290.4953613, 327.1592407, -392.1834412, 437.8486633, -726.6799316, 716.3499146
2: -231.4824219, 322.7980652, -311.5800476, 432.9999695, -663.5839844, 633.3868408
3: -325.6855774, 409.8242798, -437.9827271, 547.1330566, -871.7684326, 847.4039307
4: -296.4940491, 437.8014526, -397.5658264, 587.9273071, -883.1232300, 834.3132935

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0330176
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0330714
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -344.5086060, 405.5797424, -574.3182373, 666.1821899, -1008.5588379, 976.6859131
1: -290.4953613, 327.1592407, -480.2474365, 538.1810913, -826.3351440, 804.2877808
2: -231.4824219, 322.7980652, -384.5292053, 529.7124023, -760.0213013, 705.7310791
3: -325.6855774, 409.8242798, -535.6255493, 671.9807129, -995.8730469, 944.8042603
4: -296.4940491, 437.8014526, -491.0196228, 717.1672363, -1012.3363647, 926.8159180

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0330176
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299410, upper bound: 607.0330714
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -455.8928528, 530.4455566, -463.7151184, 542.6040039, -995.8882446, 990.9726562
1: -379.1902771, 428.5035095, -392.1213989, 437.7734985, -814.9873657, 816.7898560
2: -305.0424805, 420.5095520, -311.5290527, 432.9244690, -736.3181763, 730.6669312
3: -424.7537231, 536.0644531, -437.9073181, 547.0437622, -970.3655396, 972.5947266
4: -390.8707581, 568.3905029, -397.4994507, 587.8253174, -976.2256470, 964.7292480

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0313044
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315684
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -455.8928528, 530.4455566, -574.2443848, 666.0914917, -1118.4827881, 1100.3355713
1: -379.1902771, 428.5035095, -480.1875000, 538.1085205, -914.6448364, 904.7297974
2: -305.0424805, 420.5095520, -384.4796143, 529.6389160, -832.7576904, 803.0125732
3: -424.7537231, 536.0644531, -535.5531616, 671.8953247, -1094.4741211, 1069.9985352
4: -390.8707581, 568.3905029, -490.9555359, 717.0683594, -1105.4415283, 1057.2342529

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0313044
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299669, upper bound: 607.0315685
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -332.7513123, 391.7471619, -465.3129272, 545.4910278, -877.1098022, 855.2055664
1: -280.8043213, 315.9574890, -394.0899658, 439.9681091, -719.1379395, 707.0541382
2: -223.6177216, 311.7685242, -312.8964539, 435.2904358, -658.0408936, 623.7399902
3: -314.4937134, 395.8934631, -440.1950684, 549.9586182, -863.4841309, 835.7083130
4: -286.3516235, 422.8675842, -399.3061218, 591.1704712, -876.2660522, 821.2100220

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263010, upper bound: 607.0312461
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0263010, upper bound: 607.0320717
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -334.1754150, 393.4548645, -499.2432251, 586.1152954, -919.2836304, 891.0359497
1: -281.8243713, 317.3457947, -422.4789124, 472.8091736, -753.2133789, 737.1127930
2: -224.5627747, 313.1368103, -335.9629211, 467.6750793, -691.4434814, 648.2311401
3: -315.8885498, 397.5907288, -472.8153687, 591.0562744, -906.0880737, 870.0921021
4: -287.5869141, 424.7091675, -429.0997314, 635.1739502, -921.5162354, 852.9392090

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0225363, upper bound: 607.0316197
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259044, upper bound: 607.0305503
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0259050, upper bound: 607.0314031
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -444.3683777, 516.7774658, -464.1377258, 544.1132812, -986.0102539, 977.9783936
1: -369.5958252, 417.4387817, -393.1830750, 438.8631287, -806.5610352, 806.7695923
2: -297.2931213, 409.5901794, -312.1232300, 434.2002563, -729.8804321, 720.4353027
3: -413.7331238, 522.2994385, -439.0194092, 548.6298828, -961.0362549, 959.9787598
4: -380.8741150, 553.5848999, -398.2989502, 589.6516113, -968.1105957, 950.8679810

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0214478, upper bound: 607.0282156
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261866, upper bound: 607.0272904
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0261874, upper bound: 607.0281236
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -446.8636475, 519.5258179, -500.0941772, 587.2122192, -1031.7258301, 1016.8809204
1: -371.5931702, 419.6811218, -423.2727051, 473.6780701, -843.4755859, 839.3917236
2: -298.9184570, 411.7390747, -336.5690308, 468.5473633, -765.9013672, 747.0881348
3: -415.8891602, 525.1255493, -473.5868225, 592.1733398, -1006.8534546, 997.4199829
4: -382.9318542, 556.4659424, -429.8628235, 636.3276978, -1016.8192139, 985.4127808

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0225060, upper bound: 607.0281972
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0258720, upper bound: 607.0272662
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0258729, upper bound: 607.0280960
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -434.6857910, 508.2796936, -264.2672424, 307.6107788, -740.8955688, 772.3543091
1: -368.6515198, 410.1012878, -219.2342834, 248.6295624, -614.3723145, 629.2792358
2: -292.1144104, 405.7350464, -176.6181183, 243.9576874, -535.4902344, 581.8912964
3: -410.7531128, 513.0687866, -246.9911499, 310.8741760, -721.6271973, 758.8991699
4: -372.4248962, 551.3237305, -226.8719177, 329.8817749, -702.0803223, 777.1661377

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0341699, upper bound: 607.0267856
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0341884, upper bound: 607.0286686
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -432.2255249, 506.4165344, -334.6794128, 392.5834045, -822.6206055, 839.7603760
1: -366.7958679, 408.4484558, -281.7952271, 316.5384216, -680.0114136, 688.2912598
2: -290.6377563, 404.3231201, -224.4958954, 312.0296936, -601.6012573, 627.8234863
3: -409.0304260, 510.7529297, -315.1404419, 396.8239136, -805.3054199, 824.7694702
4: -370.6231384, 549.3652344, -287.4506836, 423.1694946, -792.6145630, 835.3311768

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0326444, upper bound: 607.0267935
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0326688, upper bound: 607.0286755
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -474.3866882, 554.3996582, -273.0612183, 318.0774231, -791.2565918, 827.4057007
1: -400.7768555, 447.4536133, -226.6571960, 257.0250854, -655.2766113, 674.1108398
2: -318.5410767, 442.2250366, -182.5067596, 252.2840881, -570.3215942, 624.3168335
3: -447.5439453, 559.5278931, -255.3708649, 321.3830566, -768.9268799, 813.8659668
4: -406.4870605, 600.4409180, -234.4511566, 341.1827698, -747.5212402, 833.9207153

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325513, upper bound: 607.0272192
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325476, upper bound: 607.0262682
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -471.8579407, 552.2356567, -343.6445618, 403.1070251, -872.9892578, 894.7019043
1: -398.8506470, 445.5657349, -289.3640747, 324.9963989, -720.9038696, 733.1691284
2: -317.0009460, 440.5965576, -230.4840240, 320.3817749, -636.4035034, 670.1427002
3: -445.6733093, 556.8826904, -323.5699158, 407.4063110, -852.5899048, 879.4666748
4: -404.5545044, 598.2006836, -295.1263733, 434.4754028, -837.9307251, 891.9059448

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0311929, upper bound: 607.0272270
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0311915, upper bound: 607.0262784
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -450.6463318, 529.5029297, -473.3537903, 554.1528931, -1001.7033081, 999.7161255
1: -382.2748108, 427.2038269, -399.9033203, 447.0479126, -825.1871338, 822.9011230
2: -303.1940613, 422.9325256, -317.6713562, 441.7932434, -743.2735596, 738.9212036
3: -427.4125671, 533.7854614, -446.9842224, 558.7081909, -984.4693604, 979.2510376
4: -387.0945435, 574.5115356, -405.7490540, 599.7075195, -984.3052368, 977.7844849

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0305834, upper bound: 607.0268041
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0306040, upper bound: 607.0286845
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -450.6463318, 529.5029297, -486.0836182, 569.0985107, -1016.7277832, 1012.5951538
1: -382.2748108, 427.2038269, -410.6032104, 459.0905762, -837.2949219, 833.7095947
2: -303.1940613, 422.9325256, -326.2767944, 453.7045593, -755.2266235, 747.5671387
3: -427.4125671, 533.7854614, -459.1246948, 573.7846069, -999.6187134, 991.4066772
4: -387.0945435, 574.5115356, -416.7698669, 615.9426270, -1000.6097412, 988.8533325

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0305834, upper bound: 607.0268041
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0306040, upper bound: 607.0286845
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -490.1675720, 575.1331787, -480.4207764, 562.6646118, -1049.9874268, 1052.6079102
1: -414.2161255, 464.1471558, -405.7844238, 453.8962708, -864.3925781, 865.9684448
2: -329.4793396, 459.0267639, -322.4352112, 448.5575256, -776.4061279, 779.8745117
3: -463.9006653, 579.7044067, -453.7630615, 567.2426147, -1029.5745850, 1032.0986328
4: -420.9264832, 623.0946045, -411.9116211, 608.8569946, -1027.3697510, 1032.6558838

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292908, upper bound: 607.0272378
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292932, upper bound: 607.0262904
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -490.1675720, 575.1331787, -493.0074463, 577.3566895, -1064.7628174, 1065.3664551
1: -414.2161255, 464.1471558, -416.3234863, 465.7336731, -876.2984009, 876.6389160
2: -329.4793396, 459.0267639, -330.9290161, 460.2702026, -788.1606445, 788.4091187
3: -463.9006653, 579.7044067, -465.6976013, 582.0484619, -1044.4555664, 1044.0524902
4: -420.9264832, 623.0946045, -422.7713928, 624.8284302, -1043.4057617, 1043.5676270

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292908, upper bound: 607.0272378
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292932, upper bound: 607.0262904
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -467.2190552, 545.8107910, -259.2879639, 301.9417725, -768.2207031, 805.0987549
1: -395.0217590, 440.5459595, -214.6073456, 244.0366974, -636.6134033, 655.1531982
2: -313.7718201, 435.4457397, -173.1915588, 239.4427643, -552.8057251, 608.2802734
3: -440.5917358, 551.0484009, -242.1877594, 304.9917908, -745.5834961, 792.3346558
4: -400.1983948, 591.2884521, -222.6091003, 323.7130127, -723.9113770, 813.0531616

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0316749, upper bound: 607.0228514
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0319775, upper bound: 607.0255674
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -483.4433594, 566.7893066, -471.5897827, 551.7531738, -1032.5222168, 1035.4598389
1: -408.7456970, 457.3983459, -398.3045044, 445.0042725, -850.0250244, 851.7298584
2: -324.9699707, 452.3848877, -316.3268738, 439.7029114, -763.0856934, 767.1276855
3: -457.1604614, 571.4283447, -444.9352722, 556.2901001, -1011.9135132, 1015.1032104
4: -414.9285583, 614.1328735, -403.9728699, 596.8682251, -1009.4851074, 1015.8253174

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0316749, upper bound: 607.0228648
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0319775, upper bound: 607.0255771
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -574.9246826, 667.2863159, -258.5369263, 301.0907898, -875.4600830, 925.8231201
1: -483.5065613, 538.8936768, -213.9129486, 243.3554230, -724.2705688, 752.8065186
2: -384.9220581, 531.6303101, -172.6895599, 238.7268829, -623.3644409, 703.9534912
3: -539.3588257, 673.7769775, -241.4709320, 304.1216736, -843.4804688, 914.4923096
4: -491.1677551, 721.8137207, -222.0006866, 322.7194214, -813.8871460, 942.8811035

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310266, upper bound: 607.0283800
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310071, upper bound: 607.0273587
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -592.2580566, 689.2864380, -472.7312622, 553.0758667, -1143.0539551, 1159.1787109
1: -498.1520996, 556.5603027, -399.2288513, 446.0848083, -940.3435059, 951.9366455
2: -396.9266663, 549.3826294, -317.0925598, 440.7386780, -836.1951904, 864.8711548
3: -556.8413086, 695.1441650, -446.0104675, 557.6281128, -1113.0189209, 1140.0283203
4: -506.8256531, 745.7554932, -404.9793396, 598.2695923, -1103.1440430, 1148.3399658

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310266, upper bound: 607.0283922
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0310071, upper bound: 607.0273712
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -461.4247742, 539.7830200, -387.1039734, 454.6272888, -912.8743896, 924.3757935
1: -390.3544312, 435.5823059, -327.9297180, 365.7536621, -752.4514160, 759.4224243
2: -310.1057739, 430.8627625, -259.1661987, 361.2753296, -669.7420654, 688.5025024
3: -435.7236328, 544.2867432, -364.9423523, 459.1347961, -893.4898071, 907.7183228
4: -395.5617981, 585.0480347, -331.2276001, 490.7022400, -884.1634521, 914.1553955

Time for backsubstitution: 1.55 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.77 + 416.60 = 420.36 seconds
