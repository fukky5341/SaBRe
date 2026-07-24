## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 9159.82088535655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188)
1: (-5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188)
2: (-7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891)
3: (-2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812)
4: (-8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 2.11 = 3.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9164.4030869, upper bound: 9164.4030869

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8124628, upper bound: 9162.6285373
time: 0.83 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6235676, upper bound: 9162.6235676
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -9163.8124628, upper bound: 9162.6285373
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -9162.6235676, upper bound: 9162.6235676

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6298.0380859, 4681.4052734, -6418.4887695, 4774.4741211, -11072.5117188, 11099.8945312
1: -5006.5761719, 4500.5805664, -5102.3422852, 4590.1997070, -9596.7744141, 9602.9218750
2: -7271.7465820, 4893.0332031, -7409.6772461, 4993.7578125, -12265.5039062, 12302.7089844
3: -2736.1677246, 6988.3081055, -2792.5607910, 7122.9155273, -9859.0830078, 9780.8691406
4: -8079.3994141, 4853.4165039, -8232.4726562, 4953.1230469, -13032.5195312, 13085.8886719

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6235676, upper bound: 9162.6235676
time: 1.32 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6235676, upper bound: 9162.6235676
time: 0.85 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8358.9042969, 6198.2509766, -6355.7998047, 4731.3745117, -13090.2773438, 12554.0507812
1: -6636.6801758, 5959.9243164, -5052.6982422, 4548.5356445, -11185.2158203, 11012.6230469
2: -9623.8066406, 6501.3457031, -7336.9077148, 4951.2314453, -14575.0371094, 13838.2539062
3: -3659.6928711, 9251.1083984, -2771.1164551, 7052.7138672, -10712.4062500, 12022.2246094
4: -10688.8964844, 6450.9218750, -8151.5087891, 4911.5732422, -15600.4687500, 14602.4287109

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1874497, upper bound: 9162.4166451
time: 0.83 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1866323, upper bound: 9162.1866323
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -9162.6235676, upper bound: 9162.6235676
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -9162.6235676, upper bound: 9162.6235676
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -9162.1874497, upper bound: 9162.4166451
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -9162.1866323, upper bound: 9162.1866323

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6298.0380859, 4681.4052734, -6298.0380859, 4681.4052734, -10979.4433594, 10979.4433594
1: -5006.5761719, 4500.5805664, -5006.5761719, 4500.5805664, -9507.1562500, 9507.1562500
2: -7271.7465820, 4893.0332031, -7271.7465820, 4893.0332031, -12164.7792969, 12164.7792969
3: -2736.1677246, 6988.3081055, -2736.1677246, 6988.3081055, -9724.4755859, 9724.4755859
4: -8079.3994141, 4853.4165039, -8079.3994141, 4853.4165039, -12932.8164062, 12932.8164062

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7476169, upper bound: 9161.5678150
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7247121, upper bound: 9161.5637239
time: 0.84 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6298.0380859, 4681.4052734, -8358.9042969, 6198.2509766, -12496.2890625, 13040.3066406
1: -5006.5761719, 4500.5805664, -6636.6801758, 5959.9243164, -10966.5000000, 11137.2597656
2: -7271.7465820, 4893.0332031, -9623.8066406, 6501.3457031, -13773.0917969, 14516.8378906
3: -2736.1677246, 6988.3081055, -3659.6928711, 9251.1083984, -11987.2763672, 10648.0009766
4: -8079.3994141, 4853.4165039, -10688.8964844, 6450.9218750, -14530.3193359, 15542.3125000

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7476169, upper bound: 9161.5678150
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7247121, upper bound: 9161.5637239
time: 0.93 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7802.7944336, 5800.5795898, -5440.1396484, 4052.5759277, -11855.3701172, 11240.7187500
1: -6196.3139648, 5576.8666992, -4326.9882812, 3894.9760742, -10091.2890625, 9903.8544922
2: -8987.9628906, 6088.9086914, -6288.3911133, 4243.3881836, -13231.3496094, 12377.2998047
3: -3430.0246582, 8642.2343750, -2370.0886230, 6037.7158203, -9467.7392578, 11012.3232422
4: -9980.4882812, 6042.8974609, -6984.3193359, 4205.0043945, -14185.4921875, 13027.2158203

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8594049, upper bound: 9159.7581361
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7130078, upper bound: 9159.7573976
time: 0.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8304.2197266, 6158.5449219, -6204.9614258, 4621.6718750, -12925.8916016, 12363.5058594
1: -6593.4067383, 5921.7080078, -4932.4438477, 4443.4531250, -11036.8574219, 10854.1523438
2: -9560.6416016, 6459.8930664, -7161.4160156, 4837.3588867, -14398.0000000, 13621.3076172
3: -3637.1933594, 9190.6591797, -2708.8657227, 6887.1850586, -10524.3789062, 11899.5234375
4: -10618.5332031, 6409.9956055, -7956.9941406, 4799.1850586, -15417.7187500, 14366.9873047

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8583779, upper bound: 9159.7127452
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7120906, upper bound: 9159.7120910
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -9162.7476169, upper bound: 9161.5678150
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -9161.7247121, upper bound: 9161.5637239
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -9162.7476169, upper bound: 9161.5678150
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -9161.7247121, upper bound: 9161.5637239
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -9159.8594049, upper bound: 9159.7581361
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.36
Output dim: 0, lower bound: -9159.7130078, upper bound: 9159.7573976
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -9159.8583779, upper bound: 9159.7127452
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.36
Output dim: 0, lower bound: -9159.7120906, upper bound: 9159.7120910

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6171.0551758, 4584.4877930, -6232.2290039, 4631.3403320, -10802.3955078, 10816.7167969
1: -4905.5644531, 4407.1103516, -4954.2167969, 4452.3203125, -9357.8837891, 9361.3251953
2: -7125.1625977, 4789.8227539, -7195.8188477, 4839.7045898, -11964.8671875, 11985.6406250
3: -2678.3530273, 6844.5351562, -2706.3164062, 6913.9038086, -9592.2558594, 9550.8515625
4: -7916.1811523, 4751.7998047, -7994.8344727, 4800.9101562, -12717.0908203, 12746.6337891

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3790741, upper bound: 9160.7000271
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1737831, upper bound: 9160.7134338
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6652.2324219, 4953.4042969, -6111.2792969, 4539.2734375, -11191.5058594, 11064.6835938
1: -5287.2490234, 4766.0849609, -4857.9873047, 4364.8974609, -9652.1464844, 9624.0703125
2: -7675.0820312, 5170.7749023, -7056.3369141, 4743.8950195, -12418.9765625, 12227.1103516
3: -2878.7878418, 7395.5224609, -2651.1274414, 6778.2177734, -9657.0058594, 10046.6494141
4: -8525.3037109, 5120.5253906, -7839.6357422, 4705.0830078, -13230.3857422, 12960.1611328

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7090207, upper bound: 9160.6949109
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7089545, upper bound: 9160.7089545
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6171.0551758, 4584.4877930, -8308.3876953, 6159.8720703, -12330.9277344, 12892.8750000
1: -4905.5644531, 4407.1103516, -6596.6035156, 5923.0991211, -10828.6630859, 11003.7138672
2: -7125.1625977, 4789.8227539, -9565.9453125, 6460.6171875, -13585.7792969, 14355.7675781
3: -2678.3530273, 6844.5351562, -3636.6203613, 9194.3945312, -11872.7470703, 10481.1552734
4: -7916.1811523, 4751.7998047, -10624.6621094, 6410.0000000, -14326.1816406, 15376.4609375

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1493861, upper bound: 9160.7122735
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7934804, upper bound: 9160.5842987
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6652.2324219, 4953.4042969, -8156.3437500, 6042.6342773, -12694.8671875, 13109.7480469
1: -5287.2490234, 4766.0849609, -6475.3227539, 5810.9672852, -11098.2148438, 11241.4082031
2: -7675.0820312, 5170.7749023, -9389.4375000, 6338.7426758, -14013.8242188, 14560.2109375
3: -2878.7878418, 7395.5224609, -3566.4501953, 9022.3242188, -11901.1113281, 10961.9707031
4: -8525.3037109, 5120.5253906, -10427.2812500, 6288.8657227, -14814.1699219, 15547.8066406

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7077704, upper bound: 9160.7081569
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7074273, upper bound: 9160.5811871
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7638.1591797, 5674.1342773, -5299.9023438, 3948.3349609, -11586.4921875, 10974.0371094
1: -6065.5410156, 5455.6259766, -4215.7045898, 3794.7827148, -9860.3222656, 9671.3300781
2: -8798.2822266, 5955.1528320, -6126.8325195, 4133.2084961, -12931.4902344, 12081.9853516
3: -3353.4890137, 8459.2685547, -2309.2854004, 5882.5957031, -9236.0849609, 10768.5537109
4: -9770.2783203, 5908.7187500, -6804.9116211, 4096.3852539, -13866.6640625, 12713.6279297

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.6801588, upper bound: 9159.7125893
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.6801588, upper bound: 9159.7573976
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8157.0610352, 6045.4526367, -6077.5659180, 4524.1074219, -12681.1669922, 12123.0175781
1: -6476.4389648, 5813.2749023, -4831.0703125, 4349.7089844, -10826.1474609, 10644.3437500
2: -9391.0976562, 6339.5205078, -7014.2290039, 4733.3764648, -14124.4746094, 13353.7500000
3: -3567.7272949, 9027.1835938, -2651.1950684, 6744.6845703, -10312.4121094, 11678.3779297
4: -10430.7822266, 6288.3774414, -7793.5004883, 4696.5092773, -15127.2910156, 14081.8779297

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.6791534, upper bound: 9159.6791535
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.6791534, upper bound: 9159.7120910
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.52 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9162.3790741, upper bound: 9160.7000271
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9162.1737831, upper bound: 9160.7134338
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9160.7090207, upper bound: 9160.6949109
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9160.7089545, upper bound: 9160.7089545
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9162.1493861, upper bound: 9160.7122735
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9161.7934804, upper bound: 9160.5842987
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9160.7077704, upper bound: 9160.7081569
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.52
Output dim: 0, lower bound: -9160.7074273, upper bound: 9160.5811871
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 0, lower bound: -9159.6801588, upper bound: 9159.7125893
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 0, lower bound: -9159.6801588, upper bound: 9159.7573976
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 0, lower bound: -9159.6791534, upper bound: 9159.6791535
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.52
Output dim: 0, lower bound: -9159.6791534, upper bound: 9159.7120910

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5686.5600586, 4231.2788086, -5398.8720703, 4017.5825195, -9704.1416016, 9630.1503906
1: -4521.8159180, 4066.9238281, -4294.2714844, 3861.5966797, -8383.4111328, 8361.1953125
2: -6570.0712891, 4425.6860352, -6241.3247070, 4204.3862305, -10774.4570312, 10667.0107422
3: -2474.2197266, 6308.3076172, -2346.7709961, 5989.9667969, -8464.1865234, 8655.0771484
4: -7297.9897461, 4388.6025391, -6932.1215820, 4165.6743164, -11463.6640625, 11320.7246094

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.5830821, upper bound: 9158.6073330
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0633800, upper bound: 9160.3854479
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6100.3198242, 4532.5126953, -6111.5253906, 4542.5439453, -10642.8623047, 10644.0380859
1: -4848.5981445, 4357.3232422, -4857.6533203, 4367.2495117, -9215.8476562, 9214.9765625
2: -7041.8955078, 4735.7958984, -7054.7084961, 4747.4726562, -11789.3681641, 11790.5039062
3: -2648.5358887, 6766.4956055, -2655.6030273, 6780.4179688, -9428.9541016, 9422.0986328
4: -7824.3100586, 4698.4125977, -7838.6435547, 4709.8085938, -12534.1171875, 12537.0556641

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7105647
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7134339
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6139.2031250, 4579.7729492, -5276.1850586, 3922.6467285, -10061.8496094, 9855.9541016
1: -4882.6528320, 4405.8549805, -4196.6484375, 3770.5117188, -8653.1640625, 8602.5019531
2: -7089.2231445, 4782.8476562, -6099.8842773, 4103.2197266, -11192.4423828, 10882.7304688
3: -2663.0751953, 6829.2294922, -2288.2331543, 5852.1147461, -8515.1894531, 9117.4628906
4: -7871.0688477, 4737.7661133, -6774.6557617, 4065.0354004, -11936.1035156, 11512.4218750

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.6793088
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.6793088
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6579.5551758, 4901.1831055, -5971.7265625, 4439.3886719, -11018.9433594, 10872.9101562
1: -5229.4975586, 4715.7207031, -4746.4418945, 4269.1411133, -9498.6376953, 9462.1591797
2: -7591.2275391, 5116.4545898, -6894.1928711, 4640.0083008, -12231.2343750, 12010.6464844
3: -2849.0537109, 7315.1733398, -2594.2128906, 6625.8452148, -9474.8984375, 9909.3867188
4: -8432.3476562, 5066.8837891, -7660.3925781, 4602.2377930, -13034.5859375, 12727.2744141

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089544
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089545
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5686.5600586, 4231.2788086, -7465.0615234, 5548.9833984, -11235.5429688, 11696.3398438
1: -4521.8159180, 4066.9238281, -5928.5585938, 5336.7636719, -9858.5761719, 9995.4824219
2: -6570.0712891, 4425.6860352, -8602.9794922, 5825.6479492, -12395.7187500, 13028.6660156
3: -2474.2197266, 6308.3076172, -3273.5993652, 8271.7451172, -10745.9638672, 9581.9072266
4: -7297.9897461, 4388.6025391, -9552.1611328, 5774.8178711, -13072.8076172, 13940.7636719

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.0387216, upper bound: 9157.7706862
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8463850, upper bound: 9160.3831384
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6100.3198242, 4532.5126953, -8217.6591797, 6094.2294922, -12194.5488281, 12750.1718750
1: -4848.5981445, 4357.3232422, -6524.7900391, 5859.9643555, -10708.5625000, 10882.1132812
2: -7041.8955078, 4735.7958984, -9461.1132812, 6392.0839844, -13433.9794922, 14196.9091797
3: -2648.5358887, 6766.4956055, -3599.3225098, 9094.2890625, -11742.8251953, 10365.8173828
4: -7824.3100586, 4698.4125977, -10507.8769531, 6342.2958984, -14166.6054688, 15206.2890625

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0335339, upper bound: 9160.5820202
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0335339, upper bound: 9160.5842987
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6139.2031250, 4579.7729492, -7311.2084961, 5431.1201172, -11570.3222656, 11890.9794922
1: -4882.6528320, 4405.8549805, -5805.5395508, 5223.8237305, -10106.4765625, 10211.3945312
2: -7089.2231445, 4782.8476562, -8424.1347656, 5702.6762695, -12791.8994141, 13206.9794922
3: -2663.0751953, 6829.2294922, -3203.2216797, 8097.7641602, -10760.8398438, 10032.4511719
4: -7871.0688477, 4737.7661133, -9352.4326172, 5653.0678711, -13524.1367188, 14090.1982422

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6579.5551758, 4901.1831055, -8044.6738281, 5961.3691406, -12540.9238281, 12945.8554688
1: -5229.4975586, 4715.7207031, -6386.8149414, 5732.8037109, -10962.3007812, 11102.5341797
2: -7591.2275391, 5116.4545898, -9260.4726562, 6253.9106445, -13845.1367188, 14376.9277344
3: -2849.0537109, 7315.1733398, -3520.2756348, 8898.7656250, -11747.8193359, 10835.4492188
4: -8432.3476562, 5066.8837891, -10283.6904297, 6204.9907227, -14637.3378906, 15350.5742188

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.26 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.26
Output dim: 0, lower bound: -9159.5830821, upper bound: 9158.6073330
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9162.0633800, upper bound: 9160.3854479
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7105647
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7134339
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.6793088
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.6793088
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089544
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089545
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.26
Output dim: 0, lower bound: -9158.0387216, upper bound: 9157.7706862
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9161.8463850, upper bound: 9160.3831384
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9161.0335339, upper bound: 9160.5820202
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9161.0335339, upper bound: 9160.5842987
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.26
Output dim: 0, lower bound: -9160.6776137, upper bound: 9160.5811871

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5475.5141602, 4069.4680176, -5287.2446289, 3932.7348633, -9408.2490234, 9356.7128906
1: -4353.7324219, 3911.3825684, -4205.5566406, 3780.0461426, -8133.7783203, 8116.9389648
2: -6327.9370117, 4258.0366211, -6113.4501953, 4116.4926758, -10444.4296875, 10371.4863281
3: -2381.7822266, 6069.4018555, -2298.3881836, 5864.5170898, -8246.2988281, 8367.7900391
4: -7028.6542969, 4221.9550781, -6789.8232422, 4078.4885254, -11107.1425781, 11011.7783203

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3332925, upper bound: 9159.4430254
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1830771, upper bound: 9158.2365793
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5339.4643555, 3973.3225098, -6111.5253906, 4542.5439453, -9882.0068359, 10084.8466797
1: -4247.0512695, 3818.8703613, -4857.6533203, 4367.2495117, -8614.2978516, 8676.5234375
2: -6172.7690430, 4157.7558594, -7054.7084961, 4747.4726562, -10920.2402344, 11212.4638672
3: -2321.1672363, 5922.5683594, -2655.6030273, 6780.4179688, -9101.5849609, 8578.1718750
4: -6855.8618164, 4119.5634766, -7838.6435547, 4709.8085938, -11565.6689453, 11958.2050781

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7105647
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7105645
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6056.9916992, 4500.5209961, -6111.5253906, 4542.5439453, -10599.5351562, 10612.0458984
1: -4814.2382812, 4326.6323242, -4857.6533203, 4367.2495117, -9181.4843750, 9184.2851562
2: -6991.6362305, 4702.5898438, -7054.7084961, 4747.4726562, -11739.1083984, 11757.2968750
3: -2630.3791504, 6718.2563477, -2655.6030273, 6780.4179688, -9410.7968750, 9373.8593750
4: -7768.4980469, 4665.7036133, -7838.6435547, 4709.8085938, -12478.3046875, 12504.3447266

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7134338
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7134338
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -5276.1850586, 3922.6467285, -9771.1562500, 9633.8798828
1: -4652.5683594, 4191.5620117, -4196.6484375, 3770.5117188, -8423.0781250, 8388.2089844
2: -6754.8217773, 4550.6113281, -6099.8842773, 4103.2197266, -10858.0410156, 10650.4960938
3: -2527.9526367, 6502.3999023, -2288.2331543, 5852.1147461, -8380.0673828, 8790.6328125
4: -7498.1806641, 4504.6772461, -6774.6557617, 4065.0354004, -11563.2138672, 11279.3330078

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -5276.1850586, 3922.6467285, -10457.5195312, 10145.0839844
1: -5194.1127930, 4684.6015625, -4196.6484375, 3770.5117188, -8964.6230469, 8881.2500000
2: -7539.5957031, 5082.8535156, -6099.8842773, 4103.2197266, -11642.8154297, 11182.7382812
3: -2830.4941406, 7265.7309570, -2288.2331543, 5852.1147461, -8682.6064453, 9553.9638672
4: -8374.8789062, 5033.6191406, -6774.6557617, 4065.0354004, -12439.9140625, 11808.2753906

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -5971.7265625, 4439.3886719, -10287.8974609, 10329.4218750
1: -4652.5683594, 4191.5620117, -4746.4418945, 4269.1411133, -8921.7070312, 8938.0000000
2: -6754.8217773, 4550.6113281, -6894.1928711, 4640.0083008, -11394.8291016, 11444.8046875
3: -2527.9526367, 6502.3999023, -2594.2128906, 6625.8452148, -9153.7978516, 9096.6132812
4: -7498.1806641, 4504.6772461, -7660.3925781, 4602.2377930, -12100.4169922, 12165.0703125

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089542
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089544
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -5971.7265625, 4439.3886719, -10974.2607422, 10840.6279297
1: -5194.1127930, 4684.6015625, -4746.4418945, 4269.1411133, -9463.2500000, 9431.0410156
2: -7539.5957031, 5082.8535156, -6894.1928711, 4640.0083008, -12179.6035156, 11977.0468750
3: -2830.4941406, 7265.7309570, -2594.2128906, 6625.8452148, -9456.3398438, 9859.9414062
4: -8374.8789062, 5033.6191406, -7660.3925781, 4602.2377930, -12977.1162109, 12694.0107422

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7087982
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7087982
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5475.5141602, 4069.4680176, -7281.6469727, 5417.5156250, -10893.0292969, 11351.1142578
1: -4353.7324219, 3911.3825684, -5783.0756836, 5210.0805664, -9563.8125000, 9694.4570312
2: -6327.9370117, 4258.0366211, -8392.8417969, 5689.8457031, -12017.7832031, 12650.8789062
3: -2381.7822266, 6069.4018555, -3199.1364746, 8070.5947266, -10452.3769531, 9268.5380859
4: -7028.6542969, 4221.9550781, -9318.9658203, 5640.0815430, -12668.7363281, 13540.9199219

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7979676, upper bound: 9158.9754616
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7402285, upper bound: 9157.9506599
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5339.4643555, 3973.3225098, -8217.6591797, 6094.2294922, -11433.6933594, 12186.4052734
1: -4247.0512695, 3818.8703613, -6524.7900391, 5859.9643555, -10107.0156250, 10343.6591797
2: -6172.7690430, 4157.7558594, -9461.1132812, 6392.0839844, -12564.8515625, 13618.8691406
3: -2321.1672363, 5922.5683594, -3599.3225098, 9094.2890625, -11415.4560547, 9521.8906250
4: -6855.8618164, 4119.5634766, -10507.8769531, 6342.2958984, -13198.1582031, 14627.4394531

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0260683, upper bound: 9159.7186995
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0260683, upper bound: 9160.5820202
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6056.9916992, 4500.5209961, -8217.6591797, 6094.2294922, -12151.2207031, 12718.1796875
1: -4814.2382812, 4326.6323242, -6524.7900391, 5859.9643555, -10674.2021484, 10851.4218750
2: -6991.6362305, 4702.5898438, -9461.1132812, 6392.0839844, -13383.7197266, 14163.7021484
3: -2630.3791504, 6718.2563477, -3599.3225098, 9094.2890625, -11724.6679688, 10317.5791016
4: -7768.4980469, 4665.7036133, -10507.8769531, 6342.2958984, -14110.7939453, 15173.5781250

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0260683, upper bound: 9159.7209933
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0260683, upper bound: 9160.5842987
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -7311.2084961, 5431.1201172, -11279.6279297, 11668.9042969
1: -4652.5683594, 4191.5620117, -5805.5395508, 5223.8237305, -9876.3906250, 9997.1015625
2: -6754.8217773, 4550.6113281, -8424.1347656, 5702.6762695, -12457.4980469, 12974.7460938
3: -2527.9526367, 6502.3999023, -3203.2216797, 8097.7641602, -10625.7167969, 9705.6210938
4: -7498.1806641, 4504.6772461, -9352.4326172, 5653.0678711, -13151.2480469, 13857.1093750

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6723544, upper bound: 9159.6736604
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6723544, upper bound: 9160.6764253
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -7311.2084961, 5431.1201172, -11965.9921875, 12180.1093750
1: -5194.1127930, 4684.6015625, -5805.5395508, 5223.8237305, -10417.9345703, 10490.1406250
2: -7539.5957031, 5082.8535156, -8424.1347656, 5702.6762695, -13242.2714844, 13506.9873047
3: -2830.4941406, 7265.7309570, -3203.2216797, 8097.7641602, -10928.2578125, 10468.9521484
4: -8374.8789062, 5033.6191406, -9352.4326172, 5653.0678711, -14027.9472656, 14386.0517578

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6723544, upper bound: 9159.6736604
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6723544, upper bound: 9160.6764254
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -8044.6738281, 5961.3691406, -11809.8779297, 12392.5820312
1: -4652.5683594, 4191.5620117, -6386.8149414, 5732.8037109, -10385.3720703, 10575.2939453
2: -6754.8217773, 4550.6113281, -9260.4726562, 6253.9106445, -13008.7314453, 13810.9443359
3: -2527.9526367, 6502.3999023, -3520.2756348, 8898.7656250, -11426.7187500, 10022.6757812
4: -7498.1806641, 4504.6772461, -10283.6904297, 6204.9907227, -13703.1718750, 14788.3671875

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6720666, upper bound: 9159.7186192
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6720666, upper bound: 9160.5803563
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -8044.6738281, 5961.3691406, -12496.2412109, 12913.5742188
1: -5194.1127930, 4684.6015625, -6386.8149414, 5732.8037109, -10926.9160156, 11071.4160156
2: -7539.5957031, 5082.8535156, -9260.4726562, 6253.9106445, -13793.5058594, 14343.3261719
3: -2830.4941406, 7265.7309570, -3520.2756348, 8898.7656250, -11729.2597656, 10786.0039062
4: -8374.8789062, 5033.6191406, -10283.6904297, 6204.9907227, -14579.8691406, 15317.3095703

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6720666, upper bound: 9159.7182608
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6720666, upper bound: 9160.5803563
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.73 seconds
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.3332925, upper bound: 9159.4430254
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.1830771, upper bound: 9158.2365793
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7105647
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7105645
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7134338
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.1455405, upper bound: 9160.7134338
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6794746, upper bound: 9160.6949109
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089542
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7089544
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7087982
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6793088, upper bound: 9160.7087982
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 0, lower bound: -9159.7979676, upper bound: 9158.9754616
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 0, lower bound: -9159.7402285, upper bound: 9157.9506599
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.0260683, upper bound: 9159.7186995
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.0260683, upper bound: 9160.5820202
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.0260683, upper bound: 9159.7209933
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9161.0260683, upper bound: 9160.5842987
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6723544, upper bound: 9159.6736604
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6723544, upper bound: 9160.6764253
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6723544, upper bound: 9159.6736604
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6723544, upper bound: 9160.6764254
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6720666, upper bound: 9159.7186192
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6720666, upper bound: 9160.5803563
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6720666, upper bound: 9159.7182608
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -9160.6720666, upper bound: 9160.5803563

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5348.4311523, 3974.1542969, -5202.3906250, 3872.5214844, -9220.9531250, 9176.5429688
1: -4252.9204102, 3819.8332520, -4138.4750977, 3722.6540527, -7975.5737305, 7958.3085938
2: -6181.2792969, 4158.3266602, -6012.0952148, 4056.5668945, -10237.8457031, 10170.4199219
3: -2326.6857910, 5928.4472656, -2264.8825684, 5770.7797852, -8097.4658203, 8193.3300781
4: -6865.4770508, 4123.4321289, -6675.8881836, 4017.7685547, -10883.2441406, 10799.3173828

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.0214866, upper bound: 9159.4430254
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.0214866, upper bound: 9159.4430254
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5352.8369141, 3984.0646973, -5084.3188477, 3792.3642578, -9145.2001953, 9068.3837891
1: -4255.1757812, 3829.7595215, -4042.3718262, 3645.6750488, -7900.8505859, 7872.1313477
2: -6184.9423828, 4169.4399414, -5876.7729492, 3970.8369141, -10155.7792969, 10046.2128906
3: -2333.0649414, 5936.0151367, -2218.3874512, 5644.3056641, -7977.3701172, 8154.4023438
4: -6870.3496094, 4133.5932617, -6527.8339844, 3933.1882324, -10803.5380859, 10661.4257812

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.0206166, upper bound: 9158.2336812
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.0206166, upper bound: 9158.2365793
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5339.4643555, 3973.3225098, -6056.9916992, 4500.5209961, -9839.9853516, 10030.3134766
1: -4247.0512695, 3818.8703613, -4814.2382812, 4326.6323242, -8573.6835938, 8633.1074219
2: -6172.7690430, 4157.7558594, -6991.6362305, 4702.5898438, -10875.3574219, 11149.3925781
3: -2321.1672363, 5922.5683594, -2630.3791504, 6718.2563477, -9039.4238281, 8552.9472656
4: -6855.8618164, 4119.5634766, -7768.4980469, 4665.7036133, -11521.5634766, 11888.0605469

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5339.4643555, 3973.3225098, -6521.0463867, 4861.1459961, -10200.6103516, 10494.3691406
1: -4247.0512695, 3818.8703613, -5183.4223633, 4677.0102539, -8924.0585938, 9002.2919922
2: -6172.7690430, 4157.7558594, -7524.6733398, 5074.9497070, -11247.7187500, 11682.4287109
3: -2321.1672363, 5922.5683594, -2826.6457520, 7252.5444336, -9573.7119141, 8749.2138672
4: -6855.8618164, 4119.5634766, -8358.0966797, 5025.9990234, -11881.8603516, 12477.6591797

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6056.9916992, 4500.5209961, -6056.9916992, 4500.5209961, -10557.5126953, 10557.5126953
1: -4814.2382812, 4326.6323242, -4814.2382812, 4326.6323242, -9140.8691406, 9140.8701172
2: -6991.6362305, 4702.5898438, -6991.6362305, 4702.5898438, -11694.2265625, 11694.2255859
3: -2630.3791504, 6718.2563477, -2630.3791504, 6718.2563477, -9348.6357422, 9348.6357422
4: -7768.4980469, 4665.7036133, -7768.4980469, 4665.7036133, -12434.2001953, 12434.1992188

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.9644076, upper bound: 9160.6521879
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0946572, upper bound: 9160.6597903
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6056.9916992, 4500.5209961, -6521.0463867, 4861.1459961, -10918.1376953, 11021.5673828
1: -4814.2382812, 4326.6323242, -5183.4223633, 4677.0102539, -9491.2451172, 9510.0546875
2: -6991.6362305, 4702.5898438, -7524.6733398, 5074.9497070, -12066.5859375, 12227.2617188
3: -2630.3791504, 6718.2563477, -2826.6457520, 7252.5444336, -9882.9238281, 9544.9023438
4: -7768.4980469, 4665.7036133, -8358.0966797, 5025.9990234, -12794.4960938, 13023.7988281

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.9644076, upper bound: 9160.6521879
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0946572, upper bound: 9160.6597903
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -5339.0151367, 3972.9123535, -9821.4199219, 9696.7109375
1: -4652.5683594, 4191.5620117, -4246.6909180, 3818.4916992, -8471.0585938, 8438.2509766
2: -6754.8217773, 4550.6113281, -6172.2656250, 4157.3413086, -10912.1630859, 10722.8769531
3: -2527.9526367, 6502.3999023, -2320.9550781, 5922.0761719, -8450.0292969, 8823.3554688
4: -7498.1806641, 4504.6772461, -6855.3076172, 4119.1459961, -11617.3261719, 11359.9843750

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.2452680, upper bound: 9154.9072232
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9154.9038850, upper bound: 9154.9038850
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -5848.5097656, 4357.6953125, -10206.2050781, 10206.2050781
1: -4652.5683594, 4191.5620117, -4652.5683594, 4191.5620117, -8844.1279297, 8844.1279297
2: -6754.8217773, 4550.6113281, -6754.8217773, 4550.6113281, -11305.4335938, 11305.4335938
3: -2527.9526367, 6502.3999023, -2527.9526367, 6502.3999023, -9030.3525391, 9030.3525391
4: -7498.1806641, 4504.6772461, -7498.1806641, 4504.6772461, -12002.8574219, 12002.8574219

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.2452682, upper bound: 9154.9072232
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9154.9038850, upper bound: 9154.9038850
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -5339.0151367, 3972.9123535, -10507.7841797, 10207.9160156
1: -5194.1127930, 4684.6015625, -4246.6909180, 3818.4916992, -9012.6015625, 8931.2919922
2: -7539.5957031, 5082.8535156, -6172.2656250, 4157.3413086, -11696.9375000, 11255.1191406
3: -2830.4941406, 7265.7309570, -2320.9550781, 5922.0761719, -8752.5703125, 9586.6845703
4: -8374.8789062, 5033.6191406, -6855.3076172, 4119.1459961, -12494.0253906, 11888.9267578

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1917457, upper bound: 9160.6517951
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6552221, upper bound: 9160.6783157
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -5848.5097656, 4357.6953125, -10892.5683594, 10717.4091797
1: -5194.1127930, 4684.6015625, -4652.5683594, 4191.5620117, -9385.6718750, 9337.1679688
2: -7539.5957031, 5082.8535156, -6754.8217773, 4550.6113281, -12090.2070312, 11837.6757812
3: -2830.4941406, 7265.7309570, -2527.9526367, 6502.3999023, -9332.8945312, 9793.6816406
4: -8374.8789062, 5033.6191406, -7498.1806641, 4504.6772461, -12879.5566406, 12531.7998047

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1917462, upper bound: 9160.6517951
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6552221, upper bound: 9160.6783157
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -6056.9916992, 4500.5209961, -10349.0302734, 10414.6875000
1: -4652.5683594, 4191.5620117, -4814.2382812, 4326.6323242, -8979.2001953, 9005.7978516
2: -6754.8217773, 4550.6113281, -6991.6362305, 4702.5898438, -11457.4111328, 11542.2480469
3: -2527.9526367, 6502.3999023, -2630.3791504, 6718.2563477, -9246.2089844, 9132.7792969
4: -7498.1806641, 4504.6772461, -7768.4980469, 4665.7036133, -12163.8818359, 12273.1757812

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -6521.0463867, 4861.1459961, -10709.6562500, 10878.7421875
1: -4652.5683594, 4191.5620117, -5183.4223633, 4677.0102539, -9329.5751953, 9374.9824219
2: -6754.8217773, 4550.6113281, -7524.6733398, 5074.9497070, -11829.7714844, 12075.2851562
3: -2527.9526367, 6502.3999023, -2826.6457520, 7252.5444336, -9780.4970703, 9329.0458984
4: -7498.1806641, 4504.6772461, -8358.0966797, 5025.9990234, -12524.1787109, 12862.7734375

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -6056.9916992, 4500.5209961, -11035.3935547, 10925.8925781
1: -5194.1127930, 4684.6015625, -4814.2382812, 4326.6323242, -9520.7441406, 9498.8388672
2: -7539.5957031, 5082.8535156, -6991.6362305, 4702.5898438, -12242.1855469, 12074.4902344
3: -2830.4941406, 7265.7309570, -2630.3791504, 6718.2563477, -9548.7500000, 9896.1093750
4: -8374.8789062, 5033.6191406, -7768.4980469, 4665.7036133, -13040.5820312, 12802.1171875

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1924220, upper bound: 9160.6520623
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6550836, upper bound: 9160.6550837
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -6521.0463867, 4861.1459961, -11396.0195312, 11389.9472656
1: -5194.1127930, 4684.6015625, -5183.4223633, 4677.0102539, -9871.1191406, 9868.0224609
2: -7539.5957031, 5082.8535156, -7524.6733398, 5074.9497070, -12614.5449219, 12607.5263672
3: -2830.4941406, 7265.7309570, -2826.6457520, 7252.5444336, -10083.0390625, 10092.3759766
4: -8374.8789062, 5033.6191406, -8358.0966797, 5025.9990234, -13400.8779297, 13391.7158203

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1924224, upper bound: 9160.6520623
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6550836, upper bound: 9160.6550837
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5339.4643555, 3973.3225098, -8170.4921875, 6058.7373047, -11398.2001953, 12138.5371094
1: -4247.0512695, 3818.8703613, -6487.3271484, 5825.9394531, -10072.9902344, 10306.1826172
2: -6172.7690430, 4157.7558594, -9407.1347656, 6354.2421875, -12527.0097656, 13564.8906250
3: -2321.1672363, 5922.5683594, -3577.6472168, 9041.6318359, -11362.7988281, 9500.2158203
4: -6855.8618164, 4119.5634766, -10448.0498047, 6304.1855469, -13160.0468750, 14567.6123047

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5339.4643555, 3973.3225098, -8814.8632812, 6533.5449219, -11873.0078125, 12774.6894531
1: -4247.0512695, 3818.8703613, -6996.9555664, 6284.2514648, -10531.3017578, 10810.1503906
2: -6172.7690430, 4157.7558594, -10141.1152344, 6846.4956055, -13019.2626953, 14294.2275391
3: -2321.1672363, 5922.5683594, -3833.6420898, 9759.3496094, -12080.5166016, 9756.2109375
4: -6855.8618164, 4119.5634766, -11256.9599609, 6783.5244141, -13639.3837891, 15372.8310547

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6056.9916992, 4500.5209961, -8170.4921875, 6058.7373047, -12115.7285156, 12671.0126953
1: -4814.2382812, 4326.6323242, -6487.3271484, 5825.9394531, -10640.1777344, 10813.9580078
2: -6991.6362305, 4702.5898438, -9407.1347656, 6354.2421875, -13345.8769531, 14109.7246094
3: -2630.3791504, 6718.2563477, -3577.6472168, 9041.6318359, -11672.0107422, 10295.9033203
4: -7768.4980469, 4665.7036133, -10448.0498047, 6304.1855469, -14072.6835938, 15113.7519531

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8959203, upper bound: 9159.5721199
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7131766, upper bound: 9159.5723316
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6056.9916992, 4500.5209961, -8814.8632812, 6533.5449219, -12590.5361328, 13315.3847656
1: -4814.2382812, 4326.6323242, -6996.9555664, 6284.2514648, -11098.4882812, 11323.5878906
2: -6991.6362305, 4702.5898438, -10141.1152344, 6846.4956055, -13838.1318359, 14843.7050781
3: -2630.3791504, 6718.2563477, -3833.6420898, 9759.3496094, -12389.7285156, 10551.8984375
4: -7768.4980469, 4665.7036133, -11256.9599609, 6783.5244141, -14552.0195312, 15922.6611328

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8959203, upper bound: 9160.4995368
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7131766, upper bound: 9160.5041030
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -7409.5781250, 5507.3666992, -11355.8750000, 11767.2734375
1: -4652.5683594, 4191.5620117, -5884.5581055, 5296.7612305, -9949.3281250, 10076.1181641
2: -6754.8217773, 4550.6113281, -8539.4326172, 5781.5205078, -12536.3408203, 13090.0439453
3: -2527.9526367, 6502.3999023, -3248.6958008, 8209.8476562, -10737.8007812, 9751.0957031
4: -7498.1806641, 4504.6772461, -9481.6269531, 5730.7734375, -13228.9511719, 13986.3046875

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -8139.2573242, 6046.2695312, -11894.7763672, 12496.9531250
1: -4652.5683594, 4191.5620117, -6461.4291992, 5817.6831055, -10470.2490234, 10652.9912109
2: -6754.8217773, 4550.6113281, -9369.9003906, 6341.2602539, -13096.0810547, 13920.5117188
3: -2527.9526367, 6502.3999023, -3544.0375977, 9022.6132812, -11550.5664062, 10046.4375000
4: -7498.1806641, 4504.6772461, -10397.9277344, 6278.9721680, -13777.1523438, 14902.6054688

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -7409.5781250, 5507.3666992, -12042.2402344, 12278.4794922
1: -5194.1127930, 4684.6015625, -5884.5581055, 5296.7612305, -10490.8720703, 10569.1582031
2: -7539.5957031, 5082.8535156, -8539.4326172, 5781.5205078, -13321.1162109, 13622.2861328
3: -2830.4941406, 7265.7309570, -3248.6958008, 8209.8476562, -11040.3417969, 10514.4257812
4: -8374.8789062, 5033.6191406, -9481.6269531, 5730.7734375, -14105.6513672, 14515.2460938

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1857322, upper bound: 9159.5549189
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6479327, upper bound: 9159.5580418
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -8139.2573242, 6046.2695312, -12581.1406250, 13006.6484375
1: -5194.1127930, 4684.6015625, -6461.4291992, 5817.6831055, -11011.7929688, 11146.0292969
2: -7539.5957031, 5082.8535156, -9369.9003906, 6341.2602539, -13880.8554688, 14452.7539062
3: -2830.4941406, 7265.7309570, -3544.0375977, 9022.6132812, -11853.1074219, 10809.7656250
4: -8374.8789062, 5033.6191406, -10397.9277344, 6278.9721680, -14653.8515625, 15431.5468750

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1857327, upper bound: 9160.5855987
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6479327, upper bound: 9160.6328065
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -8170.4921875, 6058.7373047, -11907.2451172, 12517.7031250
1: -4652.5683594, 4191.5620117, -6487.3271484, 5825.9394531, -10478.5078125, 10675.4306641
2: -6754.8217773, 4550.6113281, -9407.1347656, 6354.2421875, -13109.0615234, 13957.1728516
3: -2527.9526367, 6502.3999023, -3577.6472168, 9041.6318359, -11569.5839844, 10080.0468750
4: -7498.1806641, 4504.6772461, -10448.0498047, 6304.1855469, -13802.3662109, 14952.7265625

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5848.5097656, 4357.6953125, -8814.8632812, 6533.5449219, -12382.0527344, 13153.8554688
1: -4652.5683594, 4191.5620117, -6996.9555664, 6284.2514648, -10936.8183594, 11179.3984375
2: -6754.8217773, 4550.6113281, -10141.1152344, 6846.4956055, -13601.3173828, 14680.4472656
3: -2527.9526367, 6502.3999023, -3833.6420898, 9759.3496094, -12286.4726562, 10336.0419922
4: -7498.1806641, 4504.6772461, -11256.9599609, 6783.5244141, -14281.7021484, 15749.7216797

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -8170.4921875, 6058.7373047, -12593.6083984, 13039.3935547
1: -5194.1127930, 4684.6015625, -6487.3271484, 5825.9394531, -11020.0517578, 11171.9257812
2: -7539.5957031, 5082.8535156, -9407.1347656, 6354.2421875, -13893.8369141, 14489.9882812
3: -2830.4941406, 7265.7309570, -3577.6472168, 9041.6318359, -11872.1259766, 10843.3750000
4: -8374.8789062, 5033.6191406, -10448.0498047, 6304.1855469, -14679.0644531, 15481.6689453

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1856777, upper bound: 9159.5682091
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6476207, upper bound: 9159.5696949
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6534.8730469, 4868.9013672, -8814.8632812, 6533.5449219, -13068.4169922, 13683.7646484
1: -5194.1127930, 4684.6015625, -6996.9555664, 6284.2514648, -11478.3623047, 11681.5556641
2: -7539.5957031, 5082.8535156, -10141.1152344, 6846.4956055, -14386.0917969, 15223.9687500
3: -2830.4941406, 7265.7309570, -3833.6420898, 9759.3496094, -12589.8437500, 11099.3710938
4: -8374.8789062, 5033.6191406, -11256.9599609, 6783.5244141, -15158.4013672, 16290.5791016

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1856782, upper bound: 9160.4975123
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6476207, upper bound: 9160.4997361
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.69 seconds
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9159.0214866, upper bound: 9159.4430254
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9159.0214866, upper bound: 9159.4430254
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9159.0206166, upper bound: 9158.2336812
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9159.0206166, upper bound: 9158.2365793
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.9644076, upper bound: 9160.6521879
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9162.0946572, upper bound: 9160.6597903
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.9644076, upper bound: 9160.6521879
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9162.0946572, upper bound: 9160.6597903
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9157.2452680, upper bound: 9154.9072232
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9154.9038850, upper bound: 9154.9038850
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9157.2452682, upper bound: 9154.9072232
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -9154.9038850, upper bound: 9154.9038850
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1917457, upper bound: 9160.6517951
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6552221, upper bound: 9160.6783157
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1917462, upper bound: 9160.6517951
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6552221, upper bound: 9160.6783157
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1924220, upper bound: 9160.6520623
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6550836, upper bound: 9160.6550837
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1924224, upper bound: 9160.6520623
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6550836, upper bound: 9160.6550837
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.8959203, upper bound: 9159.5721199
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9161.7131766, upper bound: 9159.5723316
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.8959203, upper bound: 9160.4995368
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9161.7131766, upper bound: 9160.5041030
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1857322, upper bound: 9159.5549189
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6479327, upper bound: 9159.5580418
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1857327, upper bound: 9160.5855987
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6479327, upper bound: 9160.6328065
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1856777, upper bound: 9159.5682091
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6476207, upper bound: 9159.5696949
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.1856782, upper bound: 9160.4975123
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -9160.6476207, upper bound: 9160.4997361

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5997.4584961, 4461.5122070, -5951.0502930, 4424.1484375, -10421.6074219, 10412.5625000
1: -4766.9697266, 4289.5791016, -4730.0200195, 4253.1552734, -9020.1240234, 9019.5976562
2: -6923.9013672, 4665.9394531, -6869.9536133, 4623.4360352, -11547.3359375, 11535.8925781
3: -2610.2138672, 6654.8769531, -2586.7980957, 6602.9536133, -9213.1679688, 9241.6748047
4: -7692.1035156, 4627.6147461, -7633.0712891, 4587.2612305, -12279.3642578, 12260.6855469

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0814680, upper bound: 9160.6176143
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0891767, upper bound: 9162.3188654
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6004.3110352, 4463.6674805, -6047.0156250, 4493.5366211, -10497.8476562, 10510.6835938
1: -4772.6845703, 4291.0756836, -4806.3706055, 4319.8886719, -9092.5732422, 9097.4462891
2: -6931.0610352, 4664.9633789, -6980.1640625, 4695.4570312, -11626.5175781, 11645.1240234
3: -2609.9018555, 6660.8012695, -2626.4970703, 6707.3735352, -9317.2753906, 9287.2988281
4: -7701.2290039, 4628.6059570, -7755.7543945, 4658.6733398, -12359.9013672, 12384.3593750

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8851310, upper bound: 9162.4727575
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8856312, upper bound: 9161.8859996
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5997.4584961, 4461.5122070, -6430.0878906, 4794.6088867, -10792.0673828, 10891.5996094
1: -4766.9697266, 4289.5791016, -5111.3398438, 4612.9853516, -9379.9550781, 9400.9189453
2: -6923.9013672, 4665.9394531, -7420.2900391, 5005.8828125, -11929.7841797, 12086.2275391
3: -2610.2138672, 6654.8769531, -2788.5791016, 7153.8002930, -9764.0136719, 9443.4560547
4: -7692.1035156, 4627.6147461, -8242.0166016, 4957.7514648, -12649.8554688, 12869.6308594

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8945599, upper bound: 9160.1807951
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.9644076, upper bound: 9160.6521879
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6004.3110352, 4463.6674805, -6511.0200195, 4854.1362305, -10858.4472656, 10974.6875000
1: -4772.6845703, 4291.0756836, -5175.4975586, 4670.2246094, -9442.9091797, 9466.5732422
2: -6931.0610352, 4664.9633789, -7513.1318359, 5067.7084961, -11998.7675781, 12178.0947266
3: -2609.9018555, 6660.8012695, -2822.7141113, 7241.5571289, -9851.4570312, 9483.5156250
4: -7701.2290039, 4628.6059570, -8345.2607422, 5018.9096680, -12720.1367188, 12973.8671875

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0965060, upper bound: 9160.1808855
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0946572, upper bound: 9160.6597903
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -5251.9858398, 3907.1159668, -10411.1650391, 10097.8789062
1: -5169.5468750, 4662.2280273, -4177.4091797, 3755.3681641, -8924.9150391, 8839.6347656
2: -7504.1118164, 5060.7592773, -6072.0678711, 4089.0380859, -11593.1494141, 11132.8271484
3: -2818.8969727, 7232.3164062, -2283.6181641, 5825.0566406, -8643.9531250, 9515.9335938
4: -8335.1464844, 5012.3222656, -6743.6396484, 4051.7751465, -12386.9218750, 11755.9609375

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6481.7407227, 4831.8374023, -5327.5507812, 3965.2075195, -10446.9482422, 10159.3886719
1: -5152.1132812, 4648.7148438, -4237.6406250, 3811.0388184, -8963.1503906, 8886.3544922
2: -7478.4682617, 5044.5512695, -6159.1030273, 4149.4472656, -11627.9160156, 11203.6542969
3: -2809.7058105, 7207.5600586, -2316.6735840, 5909.8178711, -8719.5234375, 9524.2333984
4: -8306.8291016, 4996.1308594, -6840.6982422, 4111.3071289, -12418.1367188, 11836.8281250

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -5768.0371094, 4297.0332031, -10801.0820312, 10613.9306641
1: -5169.5468750, 4662.2280273, -4588.7910156, 4132.8964844, -9302.4423828, 9251.0185547
2: -7504.1118164, 5060.7592773, -6662.5068359, 4487.2377930, -11991.3486328, 11723.2656250
3: -2818.8969727, 7232.3164062, -2492.9677734, 6412.3999023, -9231.2968750, 9725.2841797
4: -8335.1464844, 5012.3222656, -7395.6635742, 4442.3837891, -12777.5302734, 12407.9843750

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6481.7407227, 4831.8374023, -5836.0263672, 4349.1884766, -10830.9296875, 10667.8632812
1: -5152.1132812, 4648.7148438, -4642.6860352, 4183.3671875, -9335.4765625, 9291.3994141
2: -7478.4682617, 5044.5512695, -6740.4321289, 4541.8696289, -12020.3378906, 11784.9824219
3: -2809.7058105, 7207.5600586, -2523.2055664, 6489.0825195, -9298.7880859, 9730.7656250
4: -8306.8291016, 4996.1308594, -7482.1689453, 4496.0551758, -12802.8847656, 12478.2998047

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -5951.0502930, 4424.1484375, -10928.1972656, 10796.9433594
1: -5169.5468750, 4662.2280273, -4730.0200195, 4253.1552734, -9422.7011719, 9392.2470703
2: -7504.1118164, 5060.7592773, -6869.9536133, 4623.4360352, -12127.5478516, 11930.7128906
3: -2818.8969727, 7232.3164062, -2586.7980957, 6602.9536133, -9421.8505859, 9819.1142578
4: -8335.1464844, 5012.3222656, -7633.0712891, 4587.2612305, -12922.4072266, 12645.3935547

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1872092, upper bound: 9160.2438965
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1872092, upper bound: 9161.1800089
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6481.7407227, 4831.8374023, -6047.0156250, 4493.5366211, -10975.2763672, 10878.8535156
1: -5152.1132812, 4648.7148438, -4806.3706055, 4319.8886719, -9472.0019531, 9455.0839844
2: -7478.4682617, 5044.5512695, -6980.1640625, 4695.4570312, -12173.9257812, 12024.7128906
3: -2809.7058105, 7207.5600586, -2626.4970703, 6707.3735352, -9517.0791016, 9834.0566406
4: -8306.8291016, 4996.1308594, -7755.7543945, 4658.6733398, -12965.5019531, 12751.8828125

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2797355, upper bound: 9160.2438965
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2797355, upper bound: 9161.3143175
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -6430.0878906, 4794.6088867, -11298.6582031, 11275.9814453
1: -5169.5468750, 4662.2280273, -5111.3398438, 4612.9853516, -9782.5322266, 9773.5673828
2: -7504.1118164, 5060.7592773, -7420.2900391, 5005.8828125, -12509.9941406, 12481.0488281
3: -2818.8969727, 7232.3164062, -2788.5791016, 7153.8002930, -9972.6972656, 10020.8945312
4: -8335.1464844, 5012.3222656, -8242.0166016, 4957.7514648, -13292.8984375, 13254.3388672

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1885200, upper bound: 9160.1896717
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1885199, upper bound: 9160.6520623
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6481.7407227, 4831.8374023, -6511.0200195, 4854.1362305, -11335.8769531, 11342.8564453
1: -5152.1132812, 4648.7148438, -5175.4975586, 4670.2246094, -9822.3369141, 9824.2119141
2: -7478.4682617, 5044.5512695, -7513.1318359, 5067.7084961, -12546.1757812, 12557.6835938
3: -2809.7058105, 7207.5600586, -2822.7141113, 7241.5571289, -10051.2617188, 10030.2744141
4: -8306.8291016, 4996.1308594, -8345.2607422, 5018.9096680, -13325.7382812, 13341.3906250

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2797348, upper bound: 9160.1899938
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2797348, upper bound: 9160.6550837
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5997.4584961, 4461.5122070, -8067.9394531, 5985.0122070, -11982.4697266, 12529.4511719
1: -4766.9697266, 4289.5791016, -6406.0986328, 5754.8466797, -10521.8134766, 10695.6767578
2: -6923.9013672, 4665.9394531, -9289.4208984, 6278.2124023, -13202.1113281, 13955.3603516
3: -2610.2138672, 6654.8769531, -3536.4167480, 8929.8115234, -11540.0253906, 10191.2939453
4: -7692.1035156, 4627.6147461, -10316.9931641, 6229.3520508, -13921.4541016, 14944.6074219

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6004.3110352, 4463.6674805, -8159.9877930, 6051.3349609, -12055.6464844, 12623.6523438
1: -4772.6845703, 4291.0756836, -6479.0068359, 5818.8090820, -10591.4941406, 10770.0820312
2: -6931.0610352, 4664.9633789, -9395.0869141, 6346.6240234, -13277.6855469, 14060.0507812
3: -2609.9018555, 6660.8012695, -3573.5117188, 9030.1513672, -11640.0527344, 10234.3125000
4: -7701.2290039, 4628.6059570, -10434.6406250, 6296.6875000, -13997.9160156, 15063.2460938

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5997.4584961, 4461.5122070, -8717.3105469, 6463.4101562, -12460.8671875, 13178.1367188
1: -4766.9697266, 4289.5791016, -6919.7104492, 6216.5629883, -10983.5322266, 11209.2880859
2: -6923.9013672, 4665.9394531, -10029.3417969, 6773.8979492, -13697.7978516, 14695.2812500
3: -2610.2138672, 6654.8769531, -3794.1083984, 9653.4619141, -12263.6757812, 10448.9853516
4: -7692.1035156, 4627.6147461, -11132.8007812, 6712.0664062, -14404.1699219, 15760.4160156

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8927300, upper bound: 9160.3988084
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8957159, upper bound: 9160.4995107
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6004.3110352, 4463.6674805, -8804.3476562, 6526.0849609, -12530.3964844, 13268.0146484
1: -4772.6845703, 4291.0756836, -6988.6201172, 6277.0517578, -11049.7363281, 11279.6953125
2: -6931.0610352, 4664.9633789, -10129.0400391, 6838.8173828, -13769.8789062, 14794.0019531
3: -2609.9018555, 6660.8012695, -3829.4936523, 9747.8388672, -12357.7402344, 10490.2949219
4: -7701.2290039, 4628.6059570, -11243.5117188, 6775.9853516, -14477.2119141, 15872.1171875

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2787766, upper bound: 9160.4001259
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7283400, upper bound: 9160.5040865
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -7316.1005859, 5440.7832031, -11944.8320312, 12161.9941406
1: -5169.5468750, 4662.2280273, -5810.4072266, 5232.6318359, -10402.1787109, 10472.6347656
2: -7504.1118164, 5060.7592773, -8432.0585938, 5712.9233398, -13217.0332031, 13492.8183594
3: -2818.8969727, 7232.3164062, -3211.4838867, 8108.5595703, -10927.4550781, 10443.7998047
4: -8335.1464844, 5012.3222656, -9362.2617188, 5663.1259766, -13998.2714844, 14374.5820312

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6481.7407227, 4831.8374023, -7398.2553711, 5499.3745117, -11981.1152344, 12230.0927734
1: -5152.1132812, 4648.7148438, -5875.5937500, 5289.0468750, -10441.1591797, 10524.3085938
2: -7478.4682617, 5044.5512695, -8526.4472656, 5773.2929688, -13251.7617188, 13570.9980469
3: -2809.7058105, 7207.5600586, -3244.2297363, 8197.4414062, -11007.1464844, 10451.7900391
4: -8306.8291016, 4996.1308594, -9467.1689453, 5722.6723633, -14029.5000000, 14463.2998047

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -8044.4082031, 5978.2338867, -12482.2832031, 12882.9316406
1: -5169.5468750, 4662.2280273, -6386.2558594, 5752.1904297, -10921.7363281, 11047.3417969
2: -7504.1118164, 5060.7592773, -9261.1992188, 6271.0761719, -13775.1875000, 14321.9589844
3: -2818.8969727, 7232.3164062, -3505.7248535, 8919.8720703, -11738.7685547, 10738.0410156
4: -8335.1464844, 5012.3222656, -10277.2578125, 6209.7343750, -14544.8808594, 15289.5800781

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6481.7407227, 4831.8374023, -8128.4365234, 6038.6010742, -12520.3417969, 12959.3251953
1: -5152.1132812, 4648.7148438, -6452.8754883, 5810.2866211, -10962.3994141, 11101.5869141
2: -7478.4682617, 5044.5512695, -9357.4853516, 6333.3569336, -13811.8251953, 14402.0361328
3: -2809.7058105, 7207.5600586, -3539.7675781, 9010.7626953, -11820.4677734, 10747.3281250
4: -8306.8291016, 4996.1308594, -10384.1044922, 6271.1909180, -14578.0195312, 15380.2353516

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6504.0488281, 4845.8935547, -8067.9394531, 5985.0122070, -12489.0605469, 12913.8330078
1: -5169.5468750, 4662.2280273, -6406.0986328, 5754.8466797, -10924.3925781, 11068.3242188
2: -7504.1118164, 5060.7592773, -9289.4208984, 6278.2124023, -13782.3232422, 14350.1796875
3: -2818.8969727, 7232.3164062, -3536.4167480, 8929.8115234, -11748.7089844, 10768.7333984
4: -8335.1464844, 5012.3222656, -10316.9931641, 6229.3520508, -14564.4970703, 15329.3154297

Time for backsubstitution: 1.76 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.77 + 416.90 = 420.67 seconds
