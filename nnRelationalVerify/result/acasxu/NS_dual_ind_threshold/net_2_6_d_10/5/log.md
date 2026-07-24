## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 3897.0783163271253


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1645.1527100, 2891.7785645, -1645.1527100, 2891.7785645, -4536.9311523, 4536.9311523)
1: (-287.8174744, 365.9089050, -287.8174744, 365.9089050, -653.7263794, 653.7263794)
2: (-219.8338470, 493.6749573, -219.8338470, 493.6749573, -713.5087891, 713.5087891)
3: (-230.2796783, 647.6876831, -230.2796783, 647.6876831, -877.9673462, 877.9673462)
4: (-189.1308136, 619.7971191, -189.1308136, 619.7971191, -808.9279175, 808.9279175)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.46 + 2.01 = 4.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3897.1172875, upper bound: 3897.1172875

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172676, upper bound: 3897.1172501
time: 0.69 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172691, upper bound: 3897.1172691
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.63 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -3897.1172676, upper bound: 3897.1172501
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -3897.1172691, upper bound: 3897.1172691

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1513.6944580, 2670.4951172, -1587.7785645, 2796.2514648, -4309.9453125, 4258.2734375
1: -266.2501221, 337.2012329, -278.5071411, 353.4841614, -619.7342529, 615.7082520
2: -202.9380798, 454.4072571, -212.5106354, 476.4761658, -679.4142456, 666.9179077
3: -212.1608276, 596.2941284, -222.5131531, 625.4259033, -837.5866089, 818.8072510
4: -174.6055145, 570.3820190, -182.8400879, 598.1976929, -772.8032227, 753.2221069

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171628, upper bound: 3897.1171623
time: 0.64 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171035, upper bound: 3897.1171585
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1563.2767334, 2749.4565430, -1614.2615967, 2837.7719727, -4401.0488281, 4363.7182617
1: -273.7851257, 347.7209167, -282.5245972, 359.0404053, -632.8255005, 630.2454834
2: -208.8891296, 469.5660400, -215.6981964, 484.5813599, -693.4704590, 685.2642212
3: -219.0433960, 615.8854980, -226.0297241, 635.6947021, -854.7380371, 841.9152222
4: -179.7599030, 589.5958862, -185.5893860, 608.4074097, -788.1672974, 775.1852417

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171530, upper bound: 3897.1170983
time: 0.79 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170931, upper bound: 3897.1170931
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.06 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -3897.1171628, upper bound: 3897.1171623
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -3897.1171035, upper bound: 3897.1171585
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -3897.1171530, upper bound: 3897.1170983
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -3897.1170931, upper bound: 3897.1170931

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1495.5660400, 2639.9199219, -1531.7368164, 2701.5505371, -4197.1152344, 4171.6562500
1: -263.3067932, 333.3136597, -269.3707275, 341.4334717, -604.7402344, 602.6843262
2: -200.5875092, 449.3197021, -205.2773895, 460.6871948, -661.2745972, 654.5970459
3: -209.6193848, 589.4135132, -214.6220703, 604.1023560, -813.7216797, 804.0355225
4: -172.5944672, 563.9704590, -176.6075897, 578.2857666, -750.8801880, 740.5780029

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171002, upper bound: 3897.1171568
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171002, upper bound: 3897.1171569
time: 0.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1426.1737061, 2514.9404297, -2722.8383789, 4771.8579102, -6182.5419922, 5237.7778320
1: -250.6198578, 317.5576782, -471.8218384, 605.2399902, -855.1033325, 789.3795166
2: -191.1775208, 426.9529724, -363.9233704, 815.6561890, -1004.4001465, 790.8760986
3: -200.1616669, 561.3505249, -379.8240967, 1071.4257812, -1266.2556152, 941.1746216
4: -164.4800720, 535.9788208, -313.1504822, 1023.3980103, -1183.5007324, 849.1292114

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170537, upper bound: 3897.1171513
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170568, upper bound: 3897.1171537
time: 0.68 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1545.3884277, 2719.2160645, -1558.1634521, 2742.9787598, -4288.3671875, 4277.3793945
1: -270.8830261, 343.8777771, -273.4024658, 346.9829712, -617.8659668, 617.2802734
2: -206.5885773, 464.4921875, -208.4731598, 468.7542114, -675.3427124, 672.9653320
3: -216.5362549, 609.0713501, -218.1734619, 614.3314819, -830.8677368, 827.2448120
4: -177.7760162, 583.2023315, -179.3601990, 588.4528809, -766.2288818, 762.5625000

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170296, upper bound: 3897.1170725
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170292, upper bound: 3897.1170393
time: 0.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1470.1232910, 2584.3576660, -2756.0866699, 4828.6147461, -6291.6401367, 5340.4438477
1: -257.2636719, 326.8092346, -477.1846619, 612.4866943, -869.7503052, 803.9938354
2: -196.3366394, 440.9809265, -368.2165222, 825.6826782, -1021.2557983, 809.1972046
3: -206.0480347, 578.8567505, -384.3828430, 1084.3344727, -1286.6369629, 963.2395020
4: -168.9287262, 553.8342896, -316.8671265, 1035.9898682, -1202.3116455, 870.7014160

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170489, upper bound: 3897.1170806
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170488, upper bound: 3897.1170488
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.92 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1171002, upper bound: 3897.1171568
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1171002, upper bound: 3897.1171569
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1170537, upper bound: 3897.1171513
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1170568, upper bound: 3897.1171537
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1170296, upper bound: 3897.1170725
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1170292, upper bound: 3897.1170393
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1170489, upper bound: 3897.1170806
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 0, lower bound: -3897.1170488, upper bound: 3897.1170488

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1457.4804688, 2575.3374023, -1531.7368164, 2701.5505371, -4159.0307617, 4107.0737305
1: -257.0729980, 325.1118774, -269.3707275, 341.4334717, -598.5064697, 594.4824219
2: -195.6495819, 438.7124634, -205.2773895, 460.6871948, -656.3367920, 643.9898071
3: -204.2934113, 574.9288330, -214.6220703, 604.1023560, -808.3957520, 789.5509033
4: -168.3525391, 550.6007080, -176.6075897, 578.2857666, -746.6383057, 727.2082520

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171500, upper bound: 3897.1171531
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170372, upper bound: 3897.1171413
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2641.9680176, 4636.5615234, -1531.7368164, 2701.5505371, -5343.5175781, 6151.3535156
1: -458.9176025, 587.9248657, -269.3707275, 341.4334717, -800.3510742, 856.7982788
2: -353.6308594, 792.0981445, -205.2773895, 460.6871948, -814.3180542, 995.0803833
3: -368.6761780, 1040.4422607, -214.6220703, 604.1023560, -972.7785645, 1249.4541016
4: -304.2376404, 993.6889648, -176.6075897, 578.2857666, -882.5234375, 1165.7612305

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171500, upper bound: 3897.1171561
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170372, upper bound: 3897.1171458
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1236.6976318, 2184.9487305, -2690.3247070, 4713.4541016, -5931.3823242, 4875.2734375
1: -217.6464233, 275.7878113, -465.8832092, 597.8760986, -814.6454468, 741.6710205
2: -166.0239563, 369.2288513, -359.4887085, 805.9523926, -969.1002808, 728.7175293
3: -174.1649475, 487.7878418, -375.2603760, 1058.4737549, -1226.8907471, 863.0482178
4: -142.8343811, 463.5277710, -309.3727417, 1011.1487427, -1149.2691650, 772.9004517

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170537, upper bound: 3897.1171513
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170537, upper bound: 3897.1171513
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1400.3334961, 2466.4902344, -2704.3229980, 4737.7075195, -6118.3481445, 5170.8134766
1: -245.6006622, 311.4798279, -468.3489075, 600.9373169, -845.4168091, 779.8287354
2: -187.4675140, 418.9891663, -361.3258057, 809.8510132, -994.5053101, 780.3149414
3: -196.4606934, 550.7406006, -377.2073364, 1063.9073486, -1254.2406006, 927.9478760
4: -161.3058777, 525.9758911, -310.9308472, 1016.1105347, -1172.4356689, 836.9067383

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163502, upper bound: 3897.1168041
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170568, upper bound: 3897.1171537
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1346.2382812, 2388.8627930, -1521.8665771, 2676.9526367, -4023.1909180, 3910.7292480
1: -238.4327087, 301.3087769, -266.7609558, 338.6704712, -577.1030273, 568.0697021
2: -181.4866486, 405.1210327, -203.4673157, 458.0046997, -639.4913330, 608.5883789
3: -190.2490234, 532.7880859, -213.1819611, 599.9097900, -790.1588135, 745.9699707
4: -155.9556427, 508.5576477, -175.1307983, 574.8656616, -730.8212891, 683.6884155

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170296, upper bound: 3897.1170727
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170296, upper bound: 3897.1170727
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1508.0725098, 2648.0009766, -1536.3072510, 2701.3046875, -4209.3769531, 4184.3071289
1: -263.4040222, 334.9524231, -269.0490112, 341.7770996, -605.1810303, 604.0013428
2: -201.1571960, 452.8826599, -205.2996826, 461.9509888, -663.1081543, 658.1823730
3: -211.1942749, 593.6821899, -215.0480042, 605.3431396, -816.5374146, 808.7302246
4: -173.1356812, 568.6015625, -176.6494141, 579.8953857, -753.0310669, 745.2509155

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170292, upper bound: 3897.1170392
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170292, upper bound: 3897.1170391
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1284.7843018, 2273.6577148, -2724.6269531, 4772.1376953, -6046.5156250, 4998.2841797
1: -226.7432098, 286.8589478, -471.4780579, 605.3571777, -832.1004028, 758.3369751
2: -172.7900391, 385.3499146, -363.9273376, 816.2949829, -987.7727661, 749.2772217
3: -181.7499542, 507.4396362, -379.9761047, 1071.8049316, -1249.1380615, 887.4157715
4: -148.4550323, 483.8026428, -313.2114563, 1024.1254883, -1169.7061768, 797.0140991

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170489, upper bound: 3897.1170807
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170489, upper bound: 3897.1170807
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1440.7696533, 2529.8442383, -2737.9584961, 4795.0917969, -6224.5000000, 5267.8017578
1: -251.6450653, 319.9610901, -473.7806702, 608.2582397, -859.9031982, 793.7416992
2: -192.1825256, 431.9233398, -365.6660461, 819.9802246, -1010.9699707, 797.5893555
3: -201.8728485, 566.9395142, -381.8190918, 1076.9602051, -1274.2498779, 948.7586060
4: -165.3679810, 542.4558105, -314.6875610, 1028.8311768, -1190.9672852, 857.1433716

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164394
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170488, upper bound: 3897.1170488
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.96 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1171500, upper bound: 3897.1171531
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170372, upper bound: 3897.1171413
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1171500, upper bound: 3897.1171561
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170372, upper bound: 3897.1171458
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170537, upper bound: 3897.1171513
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170537, upper bound: 3897.1171513
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1163502, upper bound: 3897.1168041
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170568, upper bound: 3897.1171537
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170296, upper bound: 3897.1170727
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170296, upper bound: 3897.1170727
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170292, upper bound: 3897.1170392
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170292, upper bound: 3897.1170391
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170489, upper bound: 3897.1170807
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170489, upper bound: 3897.1170807
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164394
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 0, lower bound: -3897.1170488, upper bound: 3897.1170488

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1422.6175537, 2511.7741699, -1340.7666016, 2377.6855469, -3800.3032227, 3852.5405273
1: -250.7265625, 317.1708069, -237.4673309, 300.1033630, -550.8298340, 554.6381226
2: -190.7551880, 428.7628479, -180.7339478, 402.6495667, -593.4047241, 609.4967651
3: -199.4518890, 561.2890625, -189.1041870, 530.3002930, -729.7521973, 750.3932495
4: -164.1905518, 538.0367432, -155.3633728, 505.3452454, -669.5357666, 693.4001465

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171012, upper bound: 3897.1171486
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171772, upper bound: 3897.1171801
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1438.1174316, 2537.9851074, -1494.4632568, 2630.5627441, -4068.6801758, 4032.4482422
1: -253.1384277, 320.4588318, -261.9532471, 332.5758362, -585.7142334, 582.4121094
2: -192.7976685, 432.6497803, -199.8634186, 449.1099854, -641.9076538, 632.5131836
3: -201.4980774, 566.8796387, -209.2743988, 588.8002930, -790.2983398, 776.1539917
4: -165.9109955, 542.9824829, -171.9798431, 563.7247314, -729.6356812, 714.9622192

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169803, upper bound: 3897.1171241
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170397, upper bound: 3897.1171652
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2610.8081055, 4579.9223633, -1340.7666016, 2377.6855469, -4988.4936523, 5901.5903320
1: -453.1697693, 580.7934570, -237.4673309, 300.1033630, -753.2731323, 817.6279297
2: -349.3399048, 782.7186890, -180.7339478, 402.6495667, -751.9895020, 960.7214355
3: -364.3225708, 1027.9171143, -189.1041870, 530.3002930, -894.6228027, 1211.0551758
4: -300.5982056, 981.8338623, -155.3633728, 505.3452454, -805.9434814, 1132.4047852

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169591
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170052, upper bound: 3897.1169645
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2625.3320312, 4605.6831055, -1494.4632568, 2630.5627441, -5255.8940430, 6078.8300781
1: -455.7683105, 584.0323486, -261.9532471, 332.5758362, -788.3441162, 845.1676025
2: -351.2796936, 786.8660889, -199.8634186, 449.1099854, -800.3896484, 984.0346680
3: -366.3240051, 1033.6662598, -209.2743988, 588.8002930, -955.1242676, 1236.5151367
4: -302.2314453, 987.1092529, -171.9798431, 563.7247314, -865.9561768, 1153.9375000

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169317
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1236.6976318, 2184.9487305, -2610.8081055, 4579.9223633, -5796.3325195, 4795.7568359
1: -217.6464233, 275.7878113, -453.1697693, 580.7934570, -797.3949585, 728.9575806
2: -166.0239563, 369.2288513, -349.3399048, 782.7186890, -945.6607056, 718.5687256
3: -174.1649475, 487.7878418, -364.3225708, 1027.9171143, -1195.9947510, 852.1104126
4: -142.8343811, 463.5277710, -300.5982056, 981.8338623, -1119.6577148, 764.1259766

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171470
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171478
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1236.6976318, 2184.9487305, -2679.0603027, 4694.3476562, -5912.7919922, 4864.0087891
1: -217.6464233, 275.7878113, -463.8577576, 595.3598633, -812.1395874, 739.6455688
2: -166.0239563, 369.2288513, -357.9604187, 802.6179810, -965.8955688, 727.1892700
3: -174.1649475, 487.7878418, -373.7650452, 1054.1402588, -1222.6580811, 861.5528564
4: -142.8343811, 463.5277710, -308.0698547, 1006.9846191, -1145.2716064, 771.5975952

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171468
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171479
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1347.1160889, 2370.7707520, -2613.0510254, 4574.4506836, -5900.6298828, 4983.8208008
1: -236.0052032, 299.3522949, -452.4092102, 580.2064819, -814.9528809, 751.7614746
2: -180.2285309, 402.5727234, -349.0104065, 782.1347046, -959.2743530, 751.5830688
3: -189.1012726, 529.6175537, -364.5003967, 1027.5488281, -1210.1916504, 894.1179199
4: -155.0916443, 505.1522827, -300.3165588, 981.2364502, -1131.0218506, 805.4688721

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168044
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168046
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1362.6589355, 2400.0578613, -2694.1894531, 4715.7797852, -6061.0585938, 5094.2465820
1: -239.3035126, 303.2109680, -466.2626953, 598.1519165, -836.6275635, 769.4735718
2: -182.4161377, 408.0303040, -359.7101135, 806.3088379, -986.2971802, 767.7403564
3: -191.1445465, 536.3329468, -376.0354614, 1059.2470703, -1244.9230957, 912.3683472
4: -156.9490814, 512.1855469, -309.6057434, 1011.6183472, -1164.0780029, 821.7912598

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169393
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169454
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1346.2382812, 2388.8627930, -1422.6175537, 2511.7741699, -3858.0124512, 3811.4804688
1: -238.4327087, 301.3087769, -250.7265625, 317.1708069, -555.6034546, 552.0352173
2: -181.4866486, 405.1210327, -190.7551880, 428.7628479, -610.2495117, 595.8762207
3: -190.2490234, 532.7880859, -199.4518890, 561.2890625, -751.5380859, 732.2399902
4: -155.9556427, 508.5576477, -164.1905518, 538.0367432, -693.9923706, 672.7481689

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170727
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170725
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1346.2382812, 2388.8627930, -1471.9027100, 2590.4223633, -3936.6606445, 3860.7656250
1: -238.4327087, 301.3087769, -258.1998901, 327.5880127, -566.0206909, 559.5085449
2: -181.4866486, 405.1210327, -196.8101196, 443.2495117, -624.7361450, 601.9311523
3: -190.2490234, 532.7880859, -206.3186493, 580.4700928, -770.7190552, 739.1066895
4: -155.9556427, 508.5576477, -169.4275665, 556.3690796, -712.3247070, 677.9852295

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170727
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170725
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1508.0725098, 2648.0009766, -1438.1174316, 2537.9851074, -4046.0566406, 4086.1184082
1: -263.4040222, 334.9524231, -253.1384277, 320.4588318, -583.8627930, 588.0908203
2: -201.1571960, 452.8826599, -192.7976685, 432.6497803, -633.8068848, 645.6802979
3: -211.1942749, 593.6821899, -201.4980774, 566.8796387, -778.0739136, 795.1802368
4: -173.1356812, 568.6015625, -165.9109955, 542.9824829, -716.1181030, 734.5125732

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170225
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170391
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1508.0725098, 2648.0009766, -1485.3525391, 2613.0646973, -4121.1362305, 4133.3535156
1: -263.4040222, 334.9524231, -260.3228760, 330.4558411, -593.8598633, 595.2752686
2: -201.1571960, 452.8826599, -198.5040436, 446.9790344, -648.1362305, 651.3867188
3: -211.1942749, 593.6821899, -208.0648346, 585.5557861, -796.7500610, 801.7468262
4: -173.1356812, 568.6015625, -170.8294525, 561.1305542, -734.2662354, 739.4310303

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170225
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170391
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1284.7843018, 2273.6577148, -2610.8081055, 4579.9223633, -5845.6879883, 4884.4658203
1: -226.7432098, 286.8589478, -453.1697693, 580.7934570, -806.8888550, 740.0286865
2: -172.7900391, 385.3499146, -349.3399048, 782.7186890, -952.7587891, 734.6898193
3: -181.7499542, 507.4396362, -364.3225708, 1027.9171143, -1203.5340576, 871.7622070
4: -148.4550323, 483.8026428, -300.5982056, 981.8338623, -1125.5518799, 784.4008789

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170807
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170733
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1284.7843018, 2273.6577148, -2679.0603027, 4694.3476562, -5969.8935547, 4952.7177734
1: -226.7432098, 286.8589478, -463.8577576, 595.3598633, -822.1030884, 750.7166748
2: -172.7900391, 385.3499146, -357.9604187, 802.6179810, -974.3780518, 743.3103027
3: -181.7499542, 507.4396362, -373.7650452, 1054.1402588, -1231.7575684, 881.2045898
4: -148.4550323, 483.8026428, -308.0698547, 1006.9846191, -1152.9396973, 791.8724976

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170806
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170732
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1384.5001221, 2431.3806152, -2644.8281250, 4628.2690430, -6000.0678711, 5076.2089844
1: -241.9807739, 307.3585205, -457.4833069, 587.0756226, -828.9590454, 764.8417969
2: -184.7401276, 415.0257874, -353.0684814, 791.6904297, -975.0080566, 768.0941162
3: -194.1952820, 544.7141113, -368.8594666, 1039.8392334, -1229.1981201, 913.5736084
4: -158.9735107, 521.1792603, -303.8554993, 993.2274780, -1148.6926270, 825.0347290

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162404, upper bound: 3897.1164387
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164394
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1405.0954590, 2467.0666504, -2722.3784180, 4764.7241211, -6160.5566406, 5189.4448242
1: -245.4481964, 311.8828125, -470.8991089, 604.3273315, -849.7753906, 782.7819214
2: -187.3499603, 420.8977356, -363.3790894, 814.9256592, -1001.3950806, 784.2768555
3: -196.9790649, 552.6769409, -379.8412476, 1070.3325195, -1263.2600098, 932.5181885
4: -161.2600403, 528.6140747, -312.7562866, 1022.4538574, -1180.9268799, 841.3703613

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167880, upper bound: 3897.1167415
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167996, upper bound: 3897.1167999
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.01 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1171012, upper bound: 3897.1171486
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1171772, upper bound: 3897.1171801
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1169803, upper bound: 3897.1171241
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170397, upper bound: 3897.1171652
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169591
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170052, upper bound: 3897.1169645
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169317
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171470
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171478
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171468
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171479
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168044
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168046
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169393
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169454
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170727
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170725
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170727
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170725
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170225
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170391
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170225
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170391
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170807
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170733
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170806
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1170255, upper bound: 3897.1170732
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1162404, upper bound: 3897.1164387
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164394
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1167880, upper bound: 3897.1167415
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -3897.1167996, upper bound: 3897.1167999

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1325.2224121, 2337.3645020, -1284.7100830, 2275.7736816, -3600.9960938, 3622.0747070
1: -233.1854858, 294.8267212, -227.2134094, 287.1331787, -520.3186646, 522.0401611
2: -177.5456543, 398.8248596, -172.9915924, 385.3997498, -562.9454346, 571.8162842
3: -185.9189301, 522.3262939, -181.2119904, 507.6678467, -693.5866699, 703.5382690
4: -152.8603058, 500.2542419, -148.7103577, 483.7505188, -636.6107788, 648.9645386

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171012, upper bound: 3897.1171485
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171012, upper bound: 3897.1171485
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1407.0546875, 2479.4980469, -1297.6737061, 2301.5988770, -3708.6530762, 3777.1718750
1: -247.7382812, 313.1924744, -230.1030121, 290.3602295, -538.0983276, 543.2954102
2: -188.3755035, 423.5695801, -174.9384918, 390.2528381, -578.6282959, 598.5079956
3: -197.2507935, 554.6519775, -183.1666412, 513.4804077, -710.7312012, 737.8185425
4: -162.1646881, 531.5287476, -150.3714752, 489.7870789, -651.9517212, 681.9002075

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171772, upper bound: 3897.1171801
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171772, upper bound: 3897.1171801
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1342.8489990, 2366.5954590, -1434.9174805, 2525.1569824, -3868.0058594, 3801.5129395
1: -235.9036865, 298.6976929, -251.3805847, 319.0088806, -554.9125366, 550.0780640
2: -179.8514252, 403.4331360, -191.8679352, 431.0784302, -610.9298706, 595.3009033
3: -188.3333282, 529.0250854, -201.0948639, 564.9140015, -753.2472534, 730.1199341
4: -154.8014679, 505.9871216, -165.1277924, 541.0314331, -695.8328857, 671.1149292

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169803, upper bound: 3897.1171241
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169803, upper bound: 3897.1171241
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1417.9866943, 2498.1499023, -1445.4049072, 2540.6127930, -3958.5996094, 3943.5546875
1: -249.4844818, 315.5516052, -253.1230621, 321.2252808, -570.7095947, 568.6746826
2: -189.8384247, 426.2324219, -193.0216370, 434.1603088, -623.9985962, 619.2539062
3: -198.7293549, 558.6464844, -202.3053894, 569.1009521, -767.8301392, 760.9518433
4: -163.3758087, 534.8977051, -166.0986023, 544.9553833, -708.3311768, 700.9962769

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170397, upper bound: 3897.1171649
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170397, upper bound: 3897.1171652
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2466.9291992, 4332.2651367, -1256.3420410, 2228.3049316, -4695.2338867, 5570.1303711
1: -428.8702087, 549.0997314, -222.5288544, 281.0600586, -709.9302979, 770.9113159
2: -330.4379578, 739.8152466, -169.3861389, 377.0090332, -707.4470215, 906.4918213
3: -344.4378662, 972.0414429, -177.5103302, 496.7666626, -841.2045288, 1143.7353516
4: -284.3209229, 928.1114502, -145.6540680, 473.2770691, -757.5979614, 1069.0594482

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169591
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169591
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2617.9970703, 4610.4204102, -1306.8243408, 2320.4233398, -4938.4199219, 5901.2880859
1: -457.7879639, 584.0678711, -231.9158173, 292.8027954, -750.5907593, 815.9836426
2: -352.0262451, 784.8002319, -176.4214478, 392.4659729, -744.4921875, 959.3540649
3: -366.5210876, 1032.0626221, -184.4324036, 517.2008057, -883.7219238, 1212.0966797
4: -302.8091431, 984.4886475, -151.6448975, 492.6052551, -795.4144287, 1132.8270264

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169643
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170052, upper bound: 3897.1169645
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2480.1069336, 4355.9819336, -1399.9323730, 2467.0422363, -4947.1484375, 5735.7695312
1: -431.2621765, 552.0720215, -245.7922058, 311.4595337, -742.7216797, 796.8928833
2: -332.2230835, 743.6434326, -187.3897400, 420.5775757, -752.8005981, 928.3212280
3: -346.2465210, 977.3211060, -196.2750397, 551.4428711, -897.6893921, 1167.4670410
4: -285.8134460, 932.9913330, -161.2844849, 528.1227417, -813.9361572, 1089.2377930

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168191, upper bound: 3897.1169320
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169317
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2637.3967285, 4645.9863281, -1462.9812012, 2573.7224121, -5211.1191406, 6090.0087891
1: -461.4290771, 588.6774902, -256.2861328, 325.4722290, -786.9013062, 844.9635620
2: -354.7876282, 790.4353027, -195.6682892, 439.5525818, -794.3402100, 984.2517090
3: -369.2956848, 1039.9779053, -204.8541412, 576.1948242, -945.4904785, 1239.7697754
4: -305.1525269, 991.7020874, -168.3644104, 551.7154541, -856.8679810, 1156.3402100

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1252.8182373, 2223.0393066, -2610.8081055, 4579.9223633, -5813.2871094, 4833.8471680
1: -222.1013031, 280.3290710, -453.1697693, 580.7934570, -802.0734253, 733.4988403
2: -168.8762360, 376.3877258, -349.3399048, 782.7186890, -948.6934814, 725.7276001
3: -176.3085785, 495.6260986, -364.3225708, 1027.9171143, -1198.2518311, 859.9486084
4: -145.2159424, 472.3661804, -300.5982056, 981.8338623, -1122.1527100, 772.9643555

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171539
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171564
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2409.3598633, 4237.8857422, -2610.8081055, 4579.9223633, -6940.7885742, 6800.9033203
1: -419.3735352, 536.4725342, -453.1697693, 580.7934570, -994.9395752, 984.3770142
2: -323.0058289, 722.2809448, -349.3399048, 782.7186890, -1099.1483154, 1065.1041260
3: -336.8304443, 950.0098267, -364.3225708, 1027.9171143, -1355.2183838, 1304.6972656
4: -277.8039551, 906.1708984, -300.5982056, 981.8338623, -1252.2352295, 1199.2614746

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171566
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171594
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1252.8182373, 2223.0393066, -2679.0603027, 4694.3476562, -5929.7465820, 4902.0991211
1: -222.1013031, 280.3290710, -463.8577576, 595.3598633, -816.8179932, 744.1868286
2: -168.8762360, 376.3877258, -357.9604187, 802.6179810, -968.9282837, 734.3481445
3: -176.3085785, 495.6260986, -373.7650452, 1054.1402588, -1224.9151611, 869.3909912
4: -145.2159424, 472.3661804, -308.0698547, 1006.9846191, -1147.7666016, 780.4360352

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171470
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171468
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2409.3598633, 4237.8857422, -2679.0603027, 4694.3476562, -7057.2480469, 6869.9116211
1: -419.3735352, 536.4725342, -463.8577576, 595.3598633, -1009.6842041, 995.3984985
2: -323.0058289, 722.2809448, -357.9604187, 802.6179810, -1119.3830566, 1073.8109131
3: -336.8304443, 950.0098267, -373.7650452, 1054.1402588, -1381.8818359, 1314.1459961
4: -277.8039551, 906.1708984, -308.0698547, 1006.9846191, -1277.8492432, 1206.7883301

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171479
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171477
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1363.9831543, 2402.3850098, -2613.0510254, 4574.4506836, -5917.1870117, 5015.4350586
1: -239.3044891, 303.2601929, -452.4092102, 580.2064819, -818.6239014, 755.6694336
2: -182.5198822, 409.7857056, -349.0104065, 782.1347046, -961.6863403, 758.7960815
3: -191.1381073, 537.0734253, -364.5003967, 1027.5488281, -1212.3125000, 901.5738525
4: -157.0983276, 514.1385498, -300.3165588, 981.2364502, -1133.0764160, 814.4550781

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1167998
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168044
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2559.7963867, 4487.7182617, -2613.0510254, 4574.4506836, -7082.8188477, 7049.6689453
1: -444.1579285, 569.0877075, -452.4092102, 580.2064819, -1018.9472656, 1015.9140625
2: -342.3745117, 766.8507080, -349.0104065, 782.1347046, -1117.7055664, 1108.9543457
3: -357.1531372, 1007.4395142, -364.5003967, 1027.5488281, -1374.6805420, 1361.4954834
4: -294.5651855, 961.9306641, -300.3165588, 981.2364502, -1268.0167236, 1254.0933838

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168041
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168041
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1230.0394287, 2174.4135742, -2604.8286133, 4561.7915039, -5778.0654297, 4779.2421875
1: -217.1558838, 274.0198059, -451.0959473, 578.3981934, -794.7750854, 725.1156616
2: -165.2799530, 367.9352417, -347.9279175, 779.7461548, -942.6911011, 715.8631592
3: -173.2231598, 484.5630798, -363.6789246, 1024.5681152, -1192.7011719, 848.2418823
4: -142.2441254, 461.9712219, -299.4693298, 978.4367065, -1116.3109131, 761.4404907

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169393
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169393
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1509.2720947, 2677.0556641, -2663.1420898, 4660.3540039, -6151.2187500, 5340.1977539
1: -268.4664612, 338.7714233, -460.7850952, 591.2559814, -858.1197510, 799.5285645
2: -204.5427551, 449.2059631, -355.5967407, 796.8350830, -998.2249756, 804.8026733
3: -213.9720612, 595.8190308, -371.6880493, 1046.8044434, -1254.6451416, 967.5070801
4: -176.2058258, 564.1183472, -306.0624390, 999.7042847, -1170.8990479, 870.1807251

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169454
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169454
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1306.4064941, 2321.3239746, -1422.6175537, 2511.7741699, -3818.1804199, 3743.9414062
1: -231.9320374, 292.6314087, -250.7265625, 317.1708069, -549.1028442, 543.3578491
2: -176.3041840, 393.8002625, -190.7551880, 428.7628479, -605.0669556, 584.5554199
3: -184.5702820, 517.4807739, -199.4518890, 561.2890625, -745.8593140, 716.9326172
4: -151.5084076, 494.3109741, -164.1905518, 538.0367432, -689.5451660, 658.5015259

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169540, upper bound: 3897.1168987
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169807, upper bound: 3897.1169071
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2490.9360352, 4376.9023438, -1422.6175537, 2511.7741699, -5002.7099609, 5783.6821289
1: -432.5662231, 554.0590820, -250.7265625, 317.1708069, -749.7370605, 804.1840210
2: -333.4559937, 746.3837280, -190.7551880, 428.7628479, -762.2188110, 935.0006714
3: -348.2400208, 981.7138062, -199.4518890, 561.2890625, -909.5290527, 1175.6638184
4: -286.8114929, 936.5280762, -164.1905518, 538.0367432, -824.8482666, 1096.3666992

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169540, upper bound: 3897.1168987
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169807, upper bound: 3897.1169071
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1306.4064941, 2321.3239746, -1471.9027100, 2590.4223633, -3896.8286133, 3793.2265625
1: -231.9320374, 292.6314087, -258.1998901, 327.5880127, -559.5200195, 550.8312378
2: -176.3041840, 393.8002625, -196.8101196, 443.2495117, -619.5535889, 590.6103516
3: -184.5702820, 517.4807739, -206.3186493, 580.4700928, -765.0403442, 723.7992554
4: -151.5084076, 494.3109741, -169.4275665, 556.3690796, -707.8774414, 663.7385254

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168581
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168020, upper bound: 3897.1168758
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2490.9360352, 4376.9023438, -1471.9027100, 2590.4223633, -5081.3569336, 5841.3403320
1: -432.5662231, 554.0590820, -258.1998901, 327.5880127, -760.1542358, 812.2589111
2: -333.4559937, 746.3837280, -196.8101196, 443.2495117, -776.7054443, 942.5831909
3: -348.2400208, 981.7138062, -206.3186493, 580.4700928, -928.7100830, 1184.1766357
4: -286.8114929, 936.5280762, -169.4275665, 556.3690796, -843.1805420, 1103.4293213

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168582
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168020, upper bound: 3897.1168755
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1470.2717285, 2584.1577148, -1438.1174316, 2537.9851074, -4008.2568359, 4022.2751465
1: -257.2726440, 326.8358765, -253.1384277, 320.4588318, -577.7314453, 579.9743042
2: -196.2990570, 442.2652588, -192.7976685, 432.6497803, -628.9487915, 635.0629272
3: -205.9037323, 579.3170166, -201.4980774, 566.8796387, -772.7833252, 780.8150635
4: -168.9469757, 555.1984253, -165.9109955, 542.9824829, -711.9293823, 721.1093750

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170025, upper bound: 3897.1167563
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171645, upper bound: 3897.1170397
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2679.2812500, 4692.8833008, -1438.1174316, 2537.9851074, -5217.2651367, 6112.1425781
1: -463.6496582, 595.2046509, -253.1384277, 320.4588318, -784.1083984, 847.3831787
2: -357.8447571, 802.1729126, -192.7976685, 432.6497803, -790.4945068, 992.3803711
3: -373.7617798, 1053.9345703, -201.4980774, 566.8796387, -940.6414185, 1249.2655029
4: -307.9678955, 1006.5030518, -165.9109955, 542.9824829, -850.9503784, 1167.4804688

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170025, upper bound: 3897.1167564
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171646, upper bound: 3897.1170544
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1470.2717285, 2584.1577148, -1485.3525391, 2613.0646973, -4083.3364258, 4069.5102539
1: -257.2726440, 326.8358765, -260.3228760, 330.4558411, -587.7285156, 587.1586914
2: -196.2990570, 442.2652588, -198.5040436, 446.9790344, -643.2780762, 640.7692871
3: -205.9037323, 579.3170166, -208.0648346, 585.5557861, -791.4594116, 787.3816528
4: -168.9469757, 555.1984253, -170.8294525, 561.1305542, -730.0775146, 726.0277710

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165375, upper bound: 3897.1166001
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165375, upper bound: 3897.1166006
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2679.2812500, 4692.8833008, -1485.3525391, 2613.0646973, -5292.3452148, 6167.8950195
1: -463.6496582, 595.2046509, -260.3228760, 330.4558411, -794.1054688, 855.5274658
2: -357.8447571, 802.1729126, -198.5040436, 446.9790344, -804.8237915, 999.7280273
3: -373.7617798, 1053.9345703, -208.0648346, 585.5557861, -959.3175659, 1257.5073242
4: -307.9678955, 1006.5030518, -170.8294525, 561.1305542, -869.0984497, 1174.3106689

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165375, upper bound: 3897.1166006
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170393
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1304.1188965, 2317.3603516, -2610.8081055, 4579.9223633, -5866.0166016, 4928.1684570
1: -231.5292664, 292.1126404, -453.1697693, 580.7934570, -811.8053589, 745.2824097
2: -175.9977875, 393.1745300, -349.3399048, 782.7186890, -956.0843506, 742.5144043
3: -184.2554474, 516.5800171, -364.3225708, 1027.9171143, -1206.2349854, 880.9024048
4: -151.2510529, 493.5224304, -300.5982056, 981.8338623, -1128.3815918, 794.1206055

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169420, upper bound: 3897.1169040
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169479, upper bound: 3897.1169026
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2490.9360352, 4376.9023438, -2610.8081055, 4579.9223633, -7023.0458984, 6942.1655273
1: -432.5662231, 554.0590820, -453.1697693, 580.7934570, -1008.5114136, 1002.1324463
2: -333.4559937, 746.3837280, -349.3399048, 782.7186890, -1109.7122803, 1089.5686035
3: -348.2400208, 981.7138062, -364.3225708, 1027.9171143, -1366.6806641, 1336.8807373
4: -286.8114929, 936.5280762, -300.5982056, 981.8338623, -1261.3369141, 1230.1229248

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169420, upper bound: 3897.1168987
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169479, upper bound: 3897.1169014
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1304.1188965, 2317.3603516, -2679.0603027, 4694.3476562, -5990.1503906, 4996.4204102
1: -231.5292664, 292.1126404, -463.8577576, 595.3598633, -826.8890381, 755.9703979
2: -175.9977875, 393.1745300, -357.9604187, 802.6179810, -977.6959839, 751.1349487
3: -184.2554474, 516.5800171, -373.7650452, 1054.1402588, -1234.4594727, 890.3447876
4: -151.2510529, 493.5224304, -308.0698547, 1006.9846191, -1155.7473145, 801.5922241

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168739
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168019, upper bound: 3897.1168865
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2490.9360352, 4376.9023438, -2679.0603027, 4694.3476562, -7148.6455078, 7020.3964844
1: -432.5662231, 554.0590820, -463.8577576, 595.3598633, -1024.1434326, 1014.0311890
2: -333.4559937, 746.3837280, -357.9604187, 802.6179810, -1131.4274902, 1099.7501221
3: -348.2400208, 981.7138062, -373.7650452, 1054.1402588, -1395.1206055, 1348.1044922
4: -286.8114929, 936.5280762, -308.0698547, 1006.9846191, -1288.7573242, 1239.4423828

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168582
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168019, upper bound: 3897.1168755
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1403.3580322, 2465.7634277, -2644.8281250, 4628.2690430, -6018.2846680, 5110.5917969
1: -245.3816223, 311.6812439, -457.4833069, 587.0756226, -832.4570312, 769.1645508
2: -187.2880707, 422.3425598, -353.0684814, 791.6904297, -977.5593872, 775.4110107
3: -196.6087036, 552.7713623, -368.8594666, 1039.8392334, -1231.6093750, 921.6308594
4: -161.1947327, 530.1020508, -303.8554993, 993.2274780, -1150.9226074, 833.9575195

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162404, upper bound: 3897.1164387
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164392
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2625.9907227, 4597.9638672, -2644.8281250, 4628.2690430, -7212.1777344, 7202.5244141
1: -454.4081116, 583.1363525, -457.4833069, 587.0756226, -1037.2683105, 1036.1730957
2: -350.6816406, 786.0394897, -353.0684814, 791.6904297, -1136.9877930, 1133.9873047
3: -366.3588867, 1032.7650146, -368.8594666, 1039.8392334, -1397.7918701, 1393.3620605
4: -301.8036194, 986.2019043, -303.8554993, 993.2274780, -1288.9089355, 1284.1645508

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162404, upper bound: 3897.1164387
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164392
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1279.3992920, 2251.7036133, -2634.4724121, 4612.5795898, -5885.5141602, 4886.1757812
1: -224.6068878, 284.0293884, -455.8329163, 584.8386230, -809.4454346, 739.8621826
2: -170.9818115, 383.4121704, -351.7486877, 788.7473755, -958.9422607, 735.1608276
3: -179.8794556, 503.8013916, -367.6924133, 1036.1334229, -1212.3096924, 871.4937134
4: -147.2033539, 481.6856995, -302.7577209, 989.7656860, -1134.3760986, 784.4432373

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167880, upper bound: 3897.1167418
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167880, upper bound: 3897.1167415
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1524.3234863, 2703.2062988, -2690.5627441, 4708.1303711, -6226.3447266, 5393.7690430
1: -270.9599609, 341.9545898, -465.3634338, 597.2843628, -867.8763428, 807.3179321
2: -206.4553070, 453.6914368, -359.1841431, 805.2331543, -1010.4334717, 812.8755493
3: -216.4882202, 601.6067505, -375.4043579, 1057.5950928, -1269.7042236, 977.0111084
4: -177.6909485, 569.9462280, -309.1419067, 1010.2482300, -1185.0150146, 879.0880737

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167999, upper bound: 3897.1167999
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167999, upper bound: 3897.1167999
time: 0.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.09 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171012, upper bound: 3897.1171485
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171012, upper bound: 3897.1171485
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171772, upper bound: 3897.1171801
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171772, upper bound: 3897.1171801
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169803, upper bound: 3897.1171241
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169803, upper bound: 3897.1171241
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170397, upper bound: 3897.1171649
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170397, upper bound: 3897.1171652
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169591
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169591
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170007, upper bound: 3897.1169643
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170052, upper bound: 3897.1169645
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168191, upper bound: 3897.1169320
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169317
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171539
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171564
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171566
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171594
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171470
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171468
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171479
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170328, upper bound: 3897.1171477
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1167998
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168044
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168041
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1163483, upper bound: 3897.1168041
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169393
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169393
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169454
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169454
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169540, upper bound: 3897.1168987
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169807, upper bound: 3897.1169071
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169540, upper bound: 3897.1168987
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169807, upper bound: 3897.1169071
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168581
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168020, upper bound: 3897.1168758
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168582
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168020, upper bound: 3897.1168755
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170025, upper bound: 3897.1167563
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171645, upper bound: 3897.1170397
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170025, upper bound: 3897.1167564
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1171646, upper bound: 3897.1170544
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1165375, upper bound: 3897.1166001
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1165375, upper bound: 3897.1166006
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1165375, upper bound: 3897.1166006
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1170225, upper bound: 3897.1170393
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169420, upper bound: 3897.1169040
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169479, upper bound: 3897.1169026
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169420, upper bound: 3897.1168987
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1169479, upper bound: 3897.1169014
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168739
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168019, upper bound: 3897.1168865
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167441, upper bound: 3897.1168582
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1168019, upper bound: 3897.1168755
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1162404, upper bound: 3897.1164387
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164392
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1162404, upper bound: 3897.1164387
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1162416, upper bound: 3897.1164392
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167880, upper bound: 3897.1167418
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167880, upper bound: 3897.1167415
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167999, upper bound: 3897.1167999
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 0, lower bound: -3897.1167999, upper bound: 3897.1167999

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1325.2224121, 2337.3645020, -1200.6411133, 2128.2919922, -3453.5144043, 3538.0053711
1: -233.1854858, 294.8267212, -212.5714874, 268.2862244, -501.4716492, 507.3981934
2: -177.5456543, 398.8248596, -161.6586609, 360.3612061, -537.9068604, 560.4833984
3: -185.9189301, 522.3262939, -168.9565430, 474.5900269, -660.5089722, 691.2828369
4: -152.8603058, 500.2542419, -139.0103607, 452.2496948, -605.1098633, 639.2645874

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169531, upper bound: 3897.1169854
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170168, upper bound: 3897.1170331
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1325.2224121, 2337.3645020, -1249.9772949, 2218.7478027, -3543.9702148, 3587.3417969
1: -233.1854858, 294.8267212, -221.6517487, 279.6127625, -512.7982178, 516.4784546
2: -177.5456543, 398.8248596, -168.5467224, 376.2088623, -553.7545166, 567.3713989
3: -185.9189301, 522.3262939, -176.6510620, 494.6040649, -680.5229492, 698.9773560
4: -152.8603058, 500.2542419, -144.8349304, 472.1626282, -625.0229492, 645.0891724

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169531, upper bound: 3897.1169854
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170168, upper bound: 3897.1170331
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1407.0546875, 2479.4980469, -1211.1843262, 2150.4689941, -3557.5231934, 3690.6823730
1: -247.7382812, 313.1924744, -215.0924072, 271.0382690, -518.7765503, 528.2847290
2: -188.3755035, 423.5695801, -163.3633423, 364.1127014, -552.4881592, 586.9326782
3: -197.2507935, 554.6519775, -170.5787506, 479.2989197, -676.5496826, 725.2305908
4: -162.1646881, 531.5287476, -140.4550323, 456.8499756, -619.0146484, 671.9836426

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170053, upper bound: 3897.1170007
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170556, upper bound: 3897.1170392
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1407.0546875, 2479.4980469, -1267.0842285, 2251.1533203, -3658.2077637, 3746.5822754
1: -247.7382812, 313.1924744, -224.9006195, 283.7058716, -531.4440308, 538.0929565
2: -188.3755035, 423.5695801, -170.7841797, 382.0456238, -570.4210815, 594.3537598
3: -197.2507935, 554.6519775, -178.8679962, 502.0405884, -699.2913208, 733.5199585
4: -162.1646881, 531.5287476, -146.7340088, 479.5392456, -641.7039185, 678.2627563

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170053, upper bound: 3897.1170006
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170556, upper bound: 3897.1170391
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1342.8489990, 2366.5954590, -1366.8765869, 2408.4514160, -3751.3002930, 3733.4721680
1: -235.9036865, 298.6976929, -239.9835205, 304.0106812, -539.9142456, 538.6810913
2: -179.8514252, 403.4331360, -182.9855957, 410.7396240, -590.5910645, 586.4186401
3: -188.3333282, 529.0250854, -191.5701904, 538.3420410, -726.6752319, 720.5952759
4: -154.8014679, 505.9871216, -157.4975891, 515.3336792, -670.1350098, 663.4847412

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167847, upper bound: 3897.1169465
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167922, upper bound: 3897.1169716
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1342.8489990, 2366.5954590, -1407.3186035, 2474.0041504, -3816.8530273, 3773.9140625
1: -235.9036865, 298.6976929, -246.2922058, 312.7066650, -548.6103516, 544.9897461
2: -179.8514252, 403.4331360, -187.9187622, 423.6211548, -603.4725952, 591.3519287
3: -188.3333282, 529.0250854, -197.1923828, 554.4940796, -742.8273926, 726.2174683
4: -154.8014679, 505.9871216, -161.7342224, 531.7199097, -686.5213013, 667.7213135

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167847, upper bound: 3897.1169465
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167922, upper bound: 3897.1169716
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1417.9866943, 2498.1499023, -1376.4649658, 2424.6628418, -3842.6494141, 3874.6147461
1: -249.4844818, 315.5516052, -241.8774719, 306.2312012, -555.7155762, 557.4290771
2: -189.8384247, 426.2324219, -184.1435394, 414.0235901, -603.8619995, 610.3759155
3: -198.7293549, 558.6464844, -192.6973572, 542.2908936, -741.0202026, 751.3438110
4: -163.3758087, 534.8977051, -158.4705963, 519.5888062, -682.9644775, 693.3681641

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168194, upper bound: 3897.1169647
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168251, upper bound: 3897.1169885
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1417.9866943, 2498.1499023, -1422.4257812, 2499.0449219, -3917.0314941, 3920.5756836
1: -249.4844818, 315.5516052, -248.6860199, 315.8562012, -565.3405151, 564.2374878
2: -189.8384247, 426.2324219, -189.7298279, 427.6630859, -617.5014038, 615.9620972
3: -198.7293549, 558.6464844, -199.1416931, 560.1767578, -758.9060669, 757.7882080
4: -163.3758087, 534.8977051, -163.3171234, 536.8735352, -700.2492065, 698.2147217

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168194, upper bound: 3897.1169647
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168194, upper bound: 3897.1169884
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2466.9291992, 4332.2651367, -1168.0563965, 2073.2517090, -4540.1806641, 5482.0664062
1: -428.8702087, 549.0997314, -207.1555634, 261.2333984, -690.1036377, 755.4480591
2: -330.4379578, 739.8152466, -157.5004730, 350.6574707, -681.0954590, 894.4821777
3: -344.4378662, 972.0414429, -164.6911011, 461.9557190, -806.3935547, 1130.9604492
4: -284.3209229, 928.1114502, -135.5006409, 440.1882019, -724.5090332, 1058.8225098

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168921, upper bound: 3897.1169460
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168933, upper bound: 3897.1169465
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2466.9291992, 4332.2651367, -1219.5288086, 2167.8879395, -4634.8168945, 5534.4375000
1: -428.8702087, 549.0997314, -216.7409973, 273.1272278, -701.9974365, 765.2842407
2: -330.4379578, 739.8152466, -164.7117310, 367.4985657, -697.9365234, 901.9451904
3: -344.4378662, 972.0414429, -172.6761169, 482.9077759, -827.3455811, 1138.9594727
4: -284.3209229, 928.1114502, -141.5406952, 461.3569336, -745.6777954, 1065.0749512

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168921, upper bound: 3897.1169457
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168933, upper bound: 3897.1169465
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2617.9970703, 4610.4204102, -1221.5991211, 2170.1569824, -4788.1523438, 5815.6484375
1: -457.7879639, 584.0678711, -216.9999695, 273.6036682, -731.3380127, 801.0678711
2: -352.0262451, 784.8002319, -164.9268494, 367.0153809, -719.0416260, 947.7322998
3: -366.5210876, 1032.0626221, -172.0236664, 483.4956665, -850.0167236, 1199.6743164
4: -302.8091431, 984.4886475, -141.7956543, 460.6375732, -763.4467163, 1122.9011230

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170016, upper bound: 3897.1169645
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170016, upper bound: 3897.1169644
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2617.9970703, 4610.4204102, -1276.4440918, 2270.2421875, -4888.2382812, 5871.9433594
1: -457.7879639, 584.0678711, -226.9567871, 286.1365662, -743.9244995, 811.0246582
2: -352.0262451, 784.8002319, -172.4691315, 384.8158875, -736.8421631, 955.5153198
3: -366.5210876, 1032.0626221, -180.4201050, 505.8807068, -872.4016113, 1208.0994873
4: -302.8091431, 984.4886475, -148.2145844, 483.0438843, -785.8530273, 1129.4848633

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170016, upper bound: 3897.1169645
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170016, upper bound: 3897.1169644
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2480.1069336, 4355.9819336, -1329.8107910, 2348.2858887, -4828.3911133, 5665.7202148
1: -431.2621765, 552.0720215, -234.2184753, 296.0719604, -727.3341064, 785.1865234
2: -332.2230835, 743.6434326, -178.3378143, 399.5796509, -731.8027344, 919.1760864
3: -346.2465210, 977.3211060, -186.5342712, 523.9625854, -870.2091064, 1157.7062988
4: -285.8134460, 932.9913330, -153.5022583, 501.6986694, -787.5119629, 1081.3928223

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168171, upper bound: 3897.1169313
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169317
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2480.1069336, 4355.9819336, -1373.7478027, 2416.5859375, -4896.6914062, 5709.6655273
1: -431.2621765, 552.0720215, -240.8445892, 305.3237915, -736.5859375, 792.1317749
2: -332.2230835, 743.6434326, -183.5592651, 413.2592163, -745.4821777, 924.6079102
3: -346.2465210, 977.3211060, -192.6191254, 541.3196411, -887.5661621, 1163.8005371
4: -285.8134460, 932.9913330, -158.0009460, 519.0140991, -804.8274536, 1086.0588379

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168171, upper bound: 3897.1169315
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168192, upper bound: 3897.1169320
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2637.3967285, 4645.9863281, -1396.3017578, 2460.8041992, -5098.2011719, 6023.0776367
1: -461.4290771, 588.6774902, -245.2362823, 310.8108215, -772.2398682, 833.9136353
2: -354.7876282, 790.4353027, -187.0124359, 419.8627625, -774.6503296, 975.4483643
3: -369.2956848, 1039.9779053, -195.5863190, 549.9988403, -919.2945557, 1230.4891357
4: -305.1525269, 991.7020874, -160.9358673, 526.9591064, -832.1116333, 1148.8375244

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169331
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168107, upper bound: 3897.1168920
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2637.3967285, 4645.9863281, -1441.0986328, 2531.2497559, -5168.6459961, 6068.2988281
1: -461.4290771, 588.6774902, -251.9677734, 320.2443237, -781.6733398, 840.6451416
2: -354.7876282, 790.4353027, -192.4134216, 433.3388672, -788.1264648, 981.0417480
3: -369.2956848, 1039.9779053, -201.8260956, 567.5626831, -936.8583984, 1236.7460938
4: -305.1525269, 991.7020874, -165.6089172, 543.9797974, -849.1323242, 1153.6489258

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168157, upper bound: 3897.1169329
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168107, upper bound: 3897.1168916
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1252.8182373, 2223.0393066, -2409.3598633, 4237.8857422, -5470.8002930, 4632.3984375
1: -222.1013031, 280.3290710, -419.3735352, 536.4725342, -757.5972290, 699.7025757
2: -168.8762360, 376.3877258, -323.0058289, 722.2809448, -888.2052002, 699.3934937
3: -176.3085785, 495.6260986, -336.8304443, 950.0098267, -1120.1324463, 832.4565430
4: -145.2159424, 472.3661804, -277.8039551, 906.1708984, -1046.3389893, 750.1701660

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171501, upper bound: 3897.1171714
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171746
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1252.8182373, 2223.0393066, -2613.6413574, 4584.0063477, -5817.1181641, 4836.6806641
1: -222.1013031, 280.3290710, -453.5584412, 581.3005371, -802.1163940, 733.8875122
2: -168.8762360, 376.3877258, -349.6288452, 783.1923218, -948.9583740, 726.0165405
3: -176.3085785, 495.6260986, -364.6692505, 1028.9039307, -1198.6362305, 860.2953491
4: -145.2159424, 472.3661804, -300.8219299, 982.4889526, -1122.2626953, 773.1881104

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171502, upper bound: 3897.1171738
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171516, upper bound: 3897.1171787
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2409.3598633, 4237.8857422, -2409.3598633, 4237.8857422, -6598.3032227, 6598.3027344
1: -419.3735352, 536.4725342, -419.3735352, 536.4725342, -950.4633789, 950.4633789
2: -323.0058289, 722.2809448, -323.0058289, 722.2809448, -1038.6599121, 1038.6597900
3: -336.8304443, 950.0098267, -336.8304443, 950.0098267, -1277.0989990, 1277.0989990
4: -277.8039551, 906.1708984, -277.8039551, 906.1708984, -1176.4217529, 1176.4217529

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170866, upper bound: 3897.1171523
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170829, upper bound: 3897.1170807
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2409.3598633, 4237.8857422, -2613.6413574, 4584.0063477, -6944.6201172, 6799.7099609
1: -419.3735352, 536.4725342, -453.5584412, 581.3005371, -994.9825439, 984.7963867
2: -323.0058289, 722.2809448, -349.6288452, 783.1923218, -1099.4132080, 1065.2052002
3: -336.8304443, 950.0098267, -364.6692505, 1028.9039307, -1355.6029053, 1304.7906494
4: -277.8039551, 906.1708984, -300.8219299, 982.4889526, -1252.3452148, 1199.3892822

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170866, upper bound: 3897.1171593
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170829, upper bound: 3897.1170888
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1252.8182373, 2223.0393066, -2490.9360352, 4376.9023438, -5612.0629883, 4713.9750977
1: -222.1013031, 280.3290710, -432.5662231, 554.0590820, -775.3527222, 712.8952637
2: -168.8762360, 376.3877258, -333.4559937, 746.3837280, -912.6694336, 709.8436890
3: -176.3085785, 495.6260986, -348.2400208, 981.7138062, -1152.3157959, 843.8660889
4: -145.2159424, 472.3661804, -286.8114929, 936.5280762, -1077.2004395, 759.1776733

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170386, upper bound: 3897.1171608
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170531, upper bound: 3897.1171651
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1252.8182373, 2223.0393066, -2679.2812500, 4692.8833008, -5927.9887695, 4902.3198242
1: -222.1013031, 280.3290710, -463.6496582, 595.2046509, -816.1889038, 743.9786987
2: -168.8762360, 376.3877258, -357.8447571, 802.1729126, -968.2678223, 734.2324829
3: -176.3085785, 495.6260986, -373.7617798, 1053.9345703, -1224.0950928, 869.3878784
4: -145.2159424, 472.3661804, -307.9678955, 1006.5030518, -1146.7525635, 780.3341064

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170386, upper bound: 3897.1171608
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170531, upper bound: 3897.1171651
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2409.3598633, 4237.8857422, -2490.9360352, 4376.9023438, -6739.5649414, 6680.5600586
1: -419.3735352, 536.4725342, -432.5662231, 554.0590820, -968.2188721, 964.0352173
2: -323.0058289, 722.2809448, -333.4559937, 746.3837280, -1063.1242676, 1049.2238770
3: -336.8304443, 950.0098267, -348.2400208, 981.7138062, -1309.2825928, 1288.5612793
4: -277.8039551, 906.1708984, -286.8114929, 936.5280762, -1207.2832031, 1185.5234375

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170536, upper bound: 3897.1171479
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170502, upper bound: 3897.1170750
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2409.3598633, 4237.8857422, -2679.2812500, 4692.8833008, -7055.4902344, 6866.0468750
1: -419.3735352, 536.4725342, -463.6496582, 595.2046509, -1009.0550537, 995.2552490
2: -323.0058289, 722.2809448, -357.8447571, 802.1729126, -1118.7224121, 1073.5007324
3: -336.8304443, 950.0098267, -373.7617798, 1053.9345703, -1381.0618896, 1313.8870850
4: -277.8039551, 906.1708984, -307.9678955, 1006.5030518, -1276.8350830, 1206.5855713

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170536, upper bound: 3897.1171479
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170501, upper bound: 3897.1170751
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1363.9831543, 2402.3850098, -2533.5617676, 4442.4267578, -5783.0175781, 4935.9467773
1: -239.3044891, 303.2601929, -439.7231750, 563.2752686, -801.5141602, 742.9833374
2: -182.5198822, 409.7857056, -338.9268494, 759.0130615, -938.3613892, 748.7125244
3: -191.1381073, 537.0734253, -353.5329590, 997.1164551, -1181.5250244, 890.6062622
4: -157.0983276, 514.1385498, -291.5793762, 952.0722656, -1103.5992432, 805.7178955

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1126537, upper bound: 3897.1154972
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1126420, upper bound: 3897.1154115
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1363.9831543, 2402.3850098, -2600.3886719, 4552.8735352, -5896.1337891, 5002.7734375
1: -239.3044891, 303.2601929, -450.0954590, 577.3863525, -815.8091431, 753.3555908
2: -182.5198822, 409.7857056, -347.2923279, 778.4132690, -958.0982056, 757.0780029
3: -191.1381073, 537.0734253, -362.8200684, 1022.6710815, -1207.5240479, 899.8934937
4: -157.0983276, 514.1385498, -298.8775330, 976.5915527, -1128.6011963, 813.0161133

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1126537, upper bound: 3897.1154972
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1126426, upper bound: 3897.1154122
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2559.7963867, 4487.7182617, -2533.5617676, 4442.4267578, -6948.6489258, 6969.4023438
1: -444.1579285, 569.0877075, -439.7231750, 563.2752686, -1001.8375244, 1003.0484009
2: -342.3745117, 766.8507080, -338.9268494, 759.0130615, -1094.3808594, 1098.7783203
3: -357.1531372, 1007.4395142, -353.5329590, 997.1164551, -1343.8931885, 1350.5048828
4: -294.5651855, 961.9306641, -291.5793762, 952.0722656, -1238.5395508, 1245.3073730

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1151228, upper bound: 3897.1155859
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1152881, upper bound: 3897.1163309
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2559.7963867, 4487.7182617, -2600.3886719, 4552.8735352, -7061.7651367, 7036.9545898
1: -444.1579285, 569.0877075, -450.0954590, 577.3863525, -1016.1325073, 1013.7907715
2: -342.3745117, 766.8507080, -347.2923279, 778.4132690, -1114.1175537, 1107.2287598
3: -357.1531372, 1007.4395142, -362.8200684, 1022.6710815, -1369.8920898, 1359.7982178
4: -294.5651855, 961.9306641, -298.8775330, 976.5915527, -1263.5416260, 1252.6613770

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1151228, upper bound: 3897.1155859
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1152881, upper bound: 3897.1163309
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1230.0394287, 2174.4135742, -2517.5607910, 4418.4648438, -5633.0556641, 4691.9746094
1: -217.1558838, 274.0198059, -437.4469299, 560.0973511, -776.2761230, 711.4667358
2: -165.2799530, 367.9352417, -336.9618835, 754.5545044, -917.2935791, 704.8970947
3: -173.2231598, 484.5630798, -351.5630798, 991.5169067, -1159.2961426, 836.1260986
4: -142.2441254, 461.9712219, -289.9671326, 946.6405640, -1084.2580566, 751.9383545

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168149, upper bound: 3897.1169393
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168149, upper bound: 3897.1169320
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1230.0394287, 2174.4135742, -2587.4128418, 4531.4770508, -5747.9633789, 4761.8261719
1: -217.1558838, 274.0198059, -447.8560486, 574.4499512, -790.7341919, 721.8757935
2: -165.2799530, 367.9352417, -345.5561523, 774.5313110, -937.5985718, 713.4913940
3: -173.2231598, 484.5630798, -361.2902527, 1017.7897949, -1185.9268799, 845.8532104
4: -142.2441254, 461.9712219, -297.4238281, 971.9543457, -1109.9835205, 759.3950195

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168149, upper bound: 3897.1169393
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168149, upper bound: 3897.1169317
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1509.2720947, 2677.0556641, -2582.3935547, 4528.2324219, -6017.7524414, 5259.4487305
1: -268.4664612, 338.7714233, -448.1587219, 574.3102417, -841.0108032, 786.7057495
2: -204.5427551, 449.2059631, -345.4428101, 773.7015381, -974.8878174, 794.6486816
3: -213.9720612, 595.8190308, -360.4716187, 1016.3510132, -1223.9084473, 956.2906494
4: -176.2058258, 564.1183472, -297.2593384, 970.5560913, -1141.4836426, 861.3776855

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168141, upper bound: 3897.1169454
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168141, upper bound: 3897.1169331
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1509.2720947, 2677.0556641, -2643.0908203, 4626.7802734, -6118.0791016, 5320.1459961
1: -268.4664612, 338.7714233, -457.4166260, 586.8453979, -853.6478271, 796.1878052
2: -204.5427551, 449.2059631, -352.9676208, 790.9428711, -992.4772339, 802.1734009
3: -213.9720612, 595.8190308, -368.9445801, 1039.1541748, -1247.0783691, 964.7634888
4: -176.2058258, 564.1183472, -303.7810364, 992.3350830, -1163.7153320, 867.8994141

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168140, upper bound: 3897.1169454
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168140, upper bound: 3897.1169328
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1219.5288086, 2167.8879395, -1272.6768799, 2251.9560547, -3471.4848633, 3440.5649414
1: -216.7409973, 273.1272278, -224.9929352, 283.6448669, -500.3858643, 498.1201782
2: -164.7117310, 367.4985657, -170.9685059, 383.0296326, -547.7413330, 538.4670410
3: -172.6761169, 482.9077759, -178.8434601, 501.9134827, -674.5895996, 661.7510986
4: -141.5406952, 461.3569336, -147.2239532, 480.9533997, -622.4940796, 608.5808716

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169971, upper bound: 3897.1169985
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169759, upper bound: 3897.1169806
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1276.4440918, 2270.2421875, -1544.0274658, 2751.6840820, -4028.1276855, 3814.2695312
1: -226.9567871, 286.1365662, -275.7272339, 347.6737366, -574.6304932, 561.8637695
2: -172.4691315, 384.8158875, -209.8388519, 463.4040222, -635.8730469, 594.6546021
3: -180.4201050, 505.8807068, -218.8459778, 612.0568237, -792.4769287, 724.7265625
4: -148.2145844, 483.0438843, -180.7624817, 581.4147949, -729.6293945, 663.8062134

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170228, upper bound: 3897.1170450
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169998, upper bound: 3897.1169887
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2404.2463379, 4226.9692383, -1272.6768799, 2251.9560547, -4656.2021484, 5486.8422852
1: -417.7347717, 534.8389282, -224.9929352, 283.6448669, -701.3796387, 759.0147095
2: -321.9813843, 720.6146851, -170.9685059, 383.0296326, -705.0109863, 889.5781860
3: -336.2489319, 947.9963379, -178.8434601, 501.9134827, -838.1623535, 1121.7624512
4: -276.9445496, 904.2949219, -147.2239532, 480.9533997, -757.8979492, 1047.3085938

Time for backsubstitution: 2.52 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.47 + 416.42 = 420.89 seconds
