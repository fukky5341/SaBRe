## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 9158.90444504786


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188)
1: (-5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188)
2: (-7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891)
3: (-2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812)
4: (-8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906)

## BASE Result
execution time: IAR + LP analysis = 1.37 + 2.11 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9164.4112857, upper bound: 9164.4112857


# Binary Search by BASE starts (time budget: 1196.53 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=11463.63671875
rel_dist={0: [-9164.409543284748, 9164.409543284746]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=11463.63671875
rel_dist={0: [-9164.405028251073, 9164.405028251073]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=11463.63671875
rel_dist={0: [-9164.397946292802, 9164.397946292804]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=11463.63671875
rel_dist={0: [-9164.392410483666, 9164.39241048367]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=11463.63671875
rel_dist={0: [-9164.388981203918, 9164.38898120392]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=11463.63671875
rel_dist={0: [-9164.387029874371, 9164.387029874371]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=11463.63671875
rel_dist={0: [-9164.38604613241, 9164.386046132408]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=11463.63671875
rel_dist={0: [-9164.38555426143, 9164.38555426143]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=11463.63671875
rel_dist={0: [-9164.385308325942, 9164.385308325942]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=11463.63671875
rel_dist={0: [-9164.385185358204, 9164.385185358202]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=11463.63671875
rel_dist={0: [-9164.385123874348, 9164.385123874348]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=11463.63671875
rel_dist={0: [-9164.385093132441, 9164.385093132441]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=11463.63671875
rel_dist={0: [-9164.385077761535, 9164.385077761537]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=11463.63671875
rel_dist={0: [-9164.385070076176, 9164.385070076176]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=11463.63671875
rel_dist={0: [-9164.385066233679, 9164.385066233677]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=11463.63671875
rel_dist={0: [-9164.385064312786, 9164.385064312788]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=11463.63671875
rel_dist={0: [-9164.385063353026, 9164.385063353024]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=11463.63671875
rel_dist={0: [-9164.385064005155, 9164.385062874397]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=11463.63671875
rel_dist={0: [-9164.385062644484, 9164.385073694357]}

## Binary Search Result
Binary search time: 74.91 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1121.62 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.4033463, upper bound: 9162.6390903
time: 0.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 0, lower bound: -9164.4033463, upper bound: 9162.6390903
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -6571.1416016, 4892.4965820, -11190.5351562, 11252.5449219
1: -5006.5761719, 4500.5805664, -5223.5683594, 4703.7553711, -9710.3320312, 9724.1484375
2: -7271.7465820, 4893.0332031, -7584.2861328, 5120.6601562, -12392.4062500, 12477.3183594
3: -2736.1677246, 6988.3081055, -2863.8518066, 7293.6987305, -10029.8662109, 9852.1601562
4: -8079.3994141, 4853.4165039, -8426.3330078, 5078.6298828, -13158.0292969, 13279.7500000

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -6530.2407227, 4861.9511719, -13220.8525391, 12728.4921875
1: -6636.6801758, 5959.9243164, -5191.0830078, 4674.3237305, -11311.0019531, 11151.0078125
2: -9623.8066406, 6501.3457031, -7537.2602539, 5088.5727539, -14712.3779297, 14038.6054688
3: -3659.6928711, 9251.1083984, -2846.3706055, 7247.9741211, -10907.6669922, 12097.4785156
4: -10688.8964844, 6450.9218750, -8374.1025391, 5047.0312500, -15735.9267578, 14825.0224609

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.73 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.06 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -6298.0380859, 4681.4052734, -10979.4433594, 10979.4433594
1: -5006.5761719, 4500.5805664, -5006.5761719, 4500.5805664, -9507.1562500, 9507.1562500
2: -7271.7465820, 4893.0332031, -7271.7465820, 4893.0332031, -12164.7792969, 12164.7792969
3: -2736.1677246, 6988.3081055, -2736.1677246, 6988.3081055, -9724.4755859, 9724.4755859
4: -8079.3994141, 4853.4165039, -8079.3994141, 4853.4165039, -12932.8164062, 12932.8164062

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4419508, upper bound: 9161.5762664
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7303934, upper bound: 9161.5676557
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -8358.9042969, 6198.2509766, -12496.2890625, 13040.3066406
1: -5006.5761719, 4500.5805664, -6636.6801758, 5959.9243164, -10966.5000000, 11137.2597656
2: -7271.7465820, 4893.0332031, -9623.8066406, 6501.3457031, -13773.0917969, 14516.8378906
3: -2736.1677246, 6988.3081055, -3659.6928711, 9251.1083984, -11987.2763672, 10648.0009766
4: -8079.3994141, 4853.4165039, -10688.8964844, 6450.9218750, -14530.3193359, 15542.3125000

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4419508, upper bound: 9161.5762664
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7303934, upper bound: 9161.5676557
time: 0.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -6298.0380859, 4681.4052734, -13040.3066406, 12496.2890625
1: -6636.6801758, 5959.9243164, -5006.5761719, 4500.5805664, -11137.2607422, 10966.5000000
2: -9623.8066406, 6501.3457031, -7271.7465820, 4893.0332031, -14516.8369141, 13773.0917969
3: -3659.6928711, 9251.1083984, -2736.1677246, 6988.3081055, -10648.0009766, 11987.2763672
4: -10688.8964844, 6450.9218750, -8079.3994141, 4853.4165039, -15542.3125000, 14530.3193359

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0965775, upper bound: 9161.5596581
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5670136, upper bound: 9161.5670139
time: 0.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -8358.9042969, 6198.2509766, -14557.1523438, 14557.1542969
1: -6636.6801758, 5959.9243164, -6636.6801758, 5959.9243164, -12596.6044922, 12596.6044922
2: -9623.8066406, 6501.3457031, -9623.8066406, 6501.3457031, -16125.1503906, 16125.1513672
3: -3659.6928711, 9251.1083984, -3659.6928711, 9251.1083984, -12910.8007812, 12910.8007812
4: -10688.8964844, 6450.9218750, -10688.8964844, 6450.9218750, -17137.2792969, 17137.2812500

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0965775, upper bound: 9161.5596581
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5670136, upper bound: 9161.5670139
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.10 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9163.4419508, upper bound: 9161.5762664
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9161.7303934, upper bound: 9161.5676557
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9163.4419508, upper bound: 9161.5762664
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9161.7303934, upper bound: 9161.5676557
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9160.0965775, upper bound: 9161.5596581
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9161.5670136, upper bound: 9161.5670139
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9160.0965775, upper bound: 9161.5596581
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -9161.5670136, upper bound: 9161.5670139

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -6298.0380859, 4681.4052734, -10852.4609375, 10882.5253906
1: -4905.5644531, 4407.1103516, -5006.5761719, 4500.5805664, -9406.1435547, 9413.6855469
2: -7125.1625977, 4789.8227539, -7271.7465820, 4893.0332031, -12018.1943359, 12061.5693359
3: -2678.3530273, 6844.5351562, -2736.1677246, 6988.3081055, -9666.6611328, 9580.7031250
4: -7916.1811523, 4751.7998047, -8079.3994141, 4853.4165039, -12769.5976562, 12831.1982422

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -6263.7612305, 4655.4848633, -11307.7167969, 11217.1660156
1: -5287.2490234, 4766.0849609, -4979.2993164, 4475.8071289, -9763.0556641, 9745.3837891
2: -7675.0820312, 5170.7749023, -7232.2177734, 4865.8159180, -12540.8964844, 12402.9912109
3: -2878.7878418, 7395.5224609, -2720.7082520, 6949.8471680, -9828.6347656, 10116.2285156
4: -8525.3037109, 5120.5253906, -8035.4130859, 4826.3886719, -13351.6914062, 13155.9375000

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8358.9042969, 6198.2509766, -12369.3066406, 12943.3916016
1: -4905.5644531, 4407.1103516, -6636.6801758, 5959.9243164, -10865.4882812, 11043.7880859
2: -7125.1625977, 4789.8227539, -9623.8066406, 6501.3457031, -13626.5078125, 14413.6269531
3: -2678.3530273, 6844.5351562, -3659.6928711, 9251.1083984, -11929.4609375, 10504.2285156
4: -7916.1811523, 4751.7998047, -10688.8964844, 6450.9218750, -14367.1015625, 15440.6943359

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7230375, upper bound: 9160.0972192
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7230375, upper bound: 9161.5676557
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -8320.8046875, 6169.1455078, -12821.3779297, 13274.2089844
1: -5287.2490234, 4766.0849609, -6606.3237305, 5932.0473633, -11219.2958984, 11372.4082031
2: -7675.0820312, 5170.7749023, -9579.6826172, 6470.9418945, -14146.0214844, 14750.4570312
3: -2878.7878418, 7395.5224609, -3642.3186035, 9208.0498047, -12086.8378906, 11037.8398438
4: -8525.3037109, 5120.5253906, -10639.6445312, 6420.6704102, -14945.9746094, 15760.1699219

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7230375, upper bound: 9160.0972192
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7230375, upper bound: 9161.5676557
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -6298.0380859, 4681.4052734, -12942.3886719, 12422.1748047
1: -6558.9648438, 5888.8247070, -5006.5761719, 4500.5805664, -11059.5449219, 10895.4003906
2: -9511.6953125, 6422.5688477, -7271.7465820, 4893.0332031, -14404.7265625, 13694.3144531
3: -3614.8420410, 9141.4384766, -2736.1677246, 6988.3081055, -10603.1503906, 11877.6064453
4: -10564.5058594, 6371.6967773, -8079.3994141, 4853.4165039, -15417.9218750, 14451.0957031

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0972192, upper bound: 9161.7230375
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0972192, upper bound: 9161.7230375
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -6263.7612305, 4655.4848633, -13560.0742188, 12862.9814453
1: -7067.9404297, 6347.4223633, -4979.2993164, 4475.8071289, -11543.7470703, 11326.7216797
2: -10244.5087891, 6914.9130859, -7232.2177734, 4865.8159180, -15110.3242188, 14147.1308594
3: -3871.0993652, 9858.5449219, -2720.7082520, 6949.8471680, -10820.9462891, 12579.2519531
4: -11372.1064453, 6851.4018555, -8035.4130859, 4826.3886719, -16198.4951172, 14886.8144531

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5676557, upper bound: 9161.7303934
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5676557, upper bound: 9161.7303934
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -8358.9042969, 6198.2509766, -14459.2343750, 14483.0400391
1: -6558.9648438, 5888.8247070, -6636.6801758, 5959.9243164, -12518.8886719, 12525.5039062
2: -9511.6953125, 6422.5688477, -9623.8066406, 6501.3457031, -16013.0410156, 16046.3720703
3: -3614.8420410, 9141.4384766, -3659.6928711, 9251.1083984, -12862.7021484, 12800.6425781
4: -10564.5058594, 6371.6967773, -10688.8964844, 6450.9218750, -17011.5996094, 17056.8359375

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0892216, upper bound: 9160.0892216
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0892216, upper bound: 9161.5596581
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -8320.8046875, 6169.1455078, -15067.1132812, 14917.9052734
1: -7067.9404297, 6347.4223633, -6606.3237305, 5932.0473633, -12999.2031250, 12953.7451172
2: -10244.5087891, 6914.9130859, -9579.6826172, 6470.9418945, -16705.8222656, 16491.4843750
3: -3871.0993652, 9858.5449219, -3642.3186035, 9208.0498047, -13079.1494141, 13484.5869141
4: -11372.1064453, 6851.4018555, -10639.6445312, 6420.6704102, -17777.7929688, 17484.3242188

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.9511890, upper bound: 9158.6648728
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.9525925, upper bound: 9158.9519551
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7310351, upper bound: 9161.7310351
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7230375, upper bound: 9160.0972192
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7230375, upper bound: 9161.5676557
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7230375, upper bound: 9160.0972192
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.7230375, upper bound: 9161.5676557
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9160.0972192, upper bound: 9161.7230375
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9160.0972192, upper bound: 9161.7230375
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.5676557, upper bound: 9161.7303934
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9161.5676557, upper bound: 9161.7303934
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9160.0892216, upper bound: 9160.0892216
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9160.0892216, upper bound: 9161.5596581
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9158.9511890, upper bound: 9158.6648728
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 0, lower bound: -9158.9525925, upper bound: 9158.9519551

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -6171.0551758, 4584.4877930, -10755.5429688, 10755.5429688
1: -4905.5644531, 4407.1103516, -4905.5644531, 4407.1103516, -9312.6718750, 9312.6718750
2: -7125.1625977, 4789.8227539, -7125.1625977, 4789.8227539, -11914.9833984, 11914.9833984
3: -2678.3530273, 6844.5351562, -2678.3530273, 6844.5351562, -9522.8886719, 9522.8886719
4: -7916.1811523, 4751.7998047, -7916.1811523, 4751.7998047, -12667.9794922, 12667.9804688

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.7209115
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1978394, upper bound: 9160.7245129
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -6652.2324219, 4953.4042969, -11124.4589844, 11236.7207031
1: -4905.5644531, 4407.1103516, -5287.2490234, 4766.0849609, -9671.6484375, 9694.3574219
2: -7125.1625977, 4789.8227539, -7675.0820312, 5170.7749023, -12295.9355469, 12464.9033203
3: -2678.3530273, 6844.5351562, -2878.7878418, 7395.5224609, -10073.8750000, 9723.3232422
4: -7916.1811523, 4751.7998047, -8525.3037109, 5120.5253906, -13036.7060547, 13277.1025391

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.7209115
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1978394, upper bound: 9160.7245129
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -6171.0551758, 4584.4877930, -11236.7207031, 11124.4589844
1: -5287.2490234, 4766.0849609, -4905.5644531, 4407.1103516, -9694.3574219, 9671.6484375
2: -7675.0820312, 5170.7749023, -7125.1625977, 4789.8227539, -12464.9033203, 12295.9355469
3: -2878.7878418, 7395.5224609, -2678.3530273, 6844.5351562, -9723.3232422, 10073.8750000
4: -8525.3037109, 5120.5253906, -7916.1811523, 4751.7998047, -13277.1015625, 13036.7060547

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6981828, upper bound: 9160.7153666
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.7132561
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -6636.2197266, 4944.3427734, -11596.5751953, 11589.6240234
1: -5287.2490234, 4766.0849609, -5274.8398438, 4757.2133789, -10044.4619141, 10040.9248047
2: -7675.0820312, 5170.7749023, -7657.7958984, 5161.5336914, -12836.6152344, 12828.5703125
3: -2878.7878418, 7395.5224609, -2874.2885742, 7380.1958008, -10258.9833984, 10269.8095703
4: -8525.3037109, 5120.5253906, -8505.7568359, 5111.6137695, -13636.9179688, 13626.2822266

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6981828, upper bound: 9160.7153666
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.7132561
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8260.9833984, 6124.1367188, -12295.1914062, 12845.4707031
1: -4905.5644531, 4407.1103516, -6558.9648438, 5888.8247070, -10794.3876953, 10966.0732422
2: -7125.1625977, 4789.8227539, -9511.6953125, 6422.5688477, -13547.7294922, 14301.5166016
3: -2678.3530273, 6844.5351562, -3614.8420410, 9141.4384766, -11819.7910156, 10459.3769531
4: -7916.1811523, 4751.7998047, -10564.5058594, 6371.6967773, -14287.8769531, 15316.3027344

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1433882, upper bound: 9159.7311895
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1912393, upper bound: 9159.7346930
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8904.5888672, 6599.2202148, -12770.2744141, 13489.0761719
1: -4905.5644531, 4407.1103516, -7067.9404297, 6347.4223633, -11252.9863281, 11475.0488281
2: -7125.1625977, 4789.8227539, -10244.5087891, 6914.9130859, -14040.0761719, 15034.3300781
3: -2678.3530273, 6844.5351562, -3871.0993652, 9858.5449219, -12536.8974609, 10715.6347656
4: -7916.1811523, 4751.7998047, -11372.1064453, 6851.4018555, -14767.5830078, 16123.9042969

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1433882, upper bound: 9160.5925181
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1912393, upper bound: 9160.5960992
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -8260.9833984, 6124.1367188, -12776.3691406, 13214.3876953
1: -5287.2490234, 4766.0849609, -6558.9648438, 5888.8247070, -11176.0732422, 11325.0498047
2: -7675.0820312, 5170.7749023, -9511.6953125, 6422.5688477, -14097.6484375, 14682.4687500
3: -2878.7878418, 7395.5224609, -3614.8420410, 9141.4384766, -12020.2265625, 11010.3642578
4: -8525.3037109, 5120.5253906, -10564.5058594, 6371.6967773, -14897.0000000, 15685.0292969

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6915827, upper bound: 9159.7256446
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7066560, upper bound: 9159.7235342
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -8904.5888672, 6599.2202148, -13251.4531250, 13857.9931641
1: -5287.2490234, 4766.0849609, -7067.9404297, 6347.4223633, -11634.6718750, 11834.0253906
2: -7675.0820312, 5170.7749023, -10244.5087891, 6914.9130859, -14589.9951172, 15415.2832031
3: -2878.7878418, 7395.5224609, -3871.0993652, 9858.5449219, -12737.3320312, 11266.6210938
4: -8525.3037109, 5120.5253906, -11372.1064453, 6851.4018555, -15376.7050781, 16492.6328125

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6915827, upper bound: 9160.5865154
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7066560, upper bound: 9160.5840084
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -6171.0551758, 4584.4877930, -12845.4707031, 12295.1914062
1: -6558.9648438, 5888.8247070, -4905.5644531, 4407.1103516, -10966.0722656, 10794.3876953
2: -9511.6953125, 6422.5688477, -7125.1625977, 4789.8227539, -14301.5175781, 13547.7294922
3: -3614.8420410, 9141.4384766, -2678.3530273, 6844.5351562, -10459.3769531, 11819.7910156
4: -10564.5058594, 6371.6967773, -7916.1811523, 4751.7998047, -15316.3027344, 14287.8769531

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.7078058
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7235341, upper bound: 9160.7066560
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -6652.2324219, 4953.4042969, -13214.3876953, 12776.3691406
1: -6558.9648438, 5888.8247070, -5287.2490234, 4766.0849609, -11325.0498047, 11176.0732422
2: -9511.6953125, 6422.5688477, -7675.0820312, 5170.7749023, -14682.4687500, 14097.6494141
3: -3614.8420410, 9141.4384766, -2878.7878418, 7395.5224609, -11010.3642578, 12020.2265625
4: -10564.5058594, 6371.6967773, -8525.3037109, 5120.5253906, -15685.0292969, 14896.9990234

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.7078058
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7235341, upper bound: 9160.7066560
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -6171.0551758, 4584.4877930, -13489.0761719, 12770.2744141
1: -7067.9404297, 6347.4223633, -4905.5644531, 4407.1103516, -11475.0488281, 11252.9863281
2: -10244.5087891, 6914.9130859, -7125.1625977, 4789.8227539, -15034.3300781, 14040.0761719
3: -3871.0993652, 9858.5449219, -2678.3530273, 6844.5351562, -10715.6347656, 12536.8984375
4: -11372.1064453, 6851.4018555, -7916.1811523, 4751.7998047, -16123.9052734, 14767.5830078

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7108860, upper bound: 9160.7145579
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.7117217
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -6636.2197266, 4944.3427734, -13848.9296875, 13235.4384766
1: -7067.9404297, 6347.4223633, -5274.8398438, 4757.2133789, -11825.1542969, 11622.2617188
2: -10244.5087891, 6914.9130859, -7657.7958984, 5161.5336914, -15406.0419922, 14572.7089844
3: -3871.0993652, 9858.5449219, -2874.2885742, 7380.1958008, -11251.2939453, 12732.8330078
4: -11372.1064453, 6851.4018555, -8505.7568359, 5111.6137695, -16483.7207031, 15357.1582031

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7108863, upper bound: 9160.7145579
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848627, upper bound: 9160.7117217
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -8260.9833984, 6124.1367188, -14385.1201172, 14385.1201172
1: -6558.9648438, 5888.8247070, -6558.9648438, 5888.8247070, -12447.7890625, 12447.7880859
2: -9511.6953125, 6422.5688477, -9511.6953125, 6422.5688477, -15933.5830078, 15933.5820312
3: -3614.8420410, 9141.4384766, -3614.8420410, 9141.4384766, -12752.3291016, 12752.3291016
4: -10564.5058594, 6371.6967773, -10564.5058594, 6371.6967773, -16931.1562500, 16931.1542969

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6707649, upper bound: 9159.7180838
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7169340, upper bound: 9159.7169341
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -8904.5888672, 6599.2202148, -14857.0205078, 15021.4072266
1: -6558.9648438, 5888.8247070, -7067.9404297, 6347.4223633, -12906.3867188, 12955.7724609
2: -9511.6953125, 6422.5688477, -10244.5087891, 6914.9130859, -16422.5664062, 16656.1679688
3: -3614.8420410, 9141.4384766, -3871.0993652, 9858.5449219, -13453.0683594, 13012.5380859
4: -10564.5058594, 6371.6967773, -11372.1064453, 6851.4018555, -17408.0605469, 17727.4003906

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6707649, upper bound: 9160.5794123
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7169340, upper bound: 9160.5782626
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -8164.8544922, 6049.4506836, -14946.4404297, 14759.9589844
1: -7067.9404297, 6347.4223633, -6482.4228516, 5817.2255859, -12884.1826172, 12829.8457031
2: -10244.5087891, 6914.9130859, -9400.0722656, 6343.6972656, -16575.8476562, 16310.1347656
3: -3871.0993652, 9858.5449219, -3569.2258301, 9034.8125000, -12905.9121094, 13409.1845703
4: -11372.1064453, 6851.4018555, -10440.6748047, 6292.2534180, -17646.5175781, 17283.2597656

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5609829, upper bound: 9158.5578086
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5609829, upper bound: 9158.5623870
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8842.5712891, 6555.7358398, -9839.7382812, 7272.6079102, -16081.4814453, 16326.5761719
1: -7018.7929688, 6305.6230469, -7801.6376953, 7000.6318359, -13997.7910156, 14070.0791016
2: -10173.1777344, 6869.6005859, -11326.9697266, 7627.3330078, -17755.4121094, 18117.1679688
3: -3846.5930176, 9791.6992188, -4267.9921875, 10881.2626953, -14689.1279297, 14019.7978516
4: -11292.7470703, 6806.6904297, -12583.8447266, 7551.2636719, -18793.4296875, 19291.9355469

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5610246, upper bound: 9158.5578091
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.5610246, upper bound: 9158.9519562
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.32 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.7209115
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9163.1978394, upper bound: 9160.7245129
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.7209115
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9163.1978394, upper bound: 9160.7245129
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.6981828, upper bound: 9160.7153666
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.7132561
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.6981828, upper bound: 9160.7153666
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.7132561
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9161.1433882, upper bound: 9159.7311895
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9163.1912393, upper bound: 9159.7346930
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9161.1433882, upper bound: 9160.5925181
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9163.1912393, upper bound: 9160.5960992
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.6915827, upper bound: 9159.7256446
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.7066560, upper bound: 9159.7235342
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.6915827, upper bound: 9160.5865154
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.7066560, upper bound: 9160.5840084
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.7078058
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.7235341, upper bound: 9160.7066560
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.7078058
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.7235341, upper bound: 9160.7066560
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.7108860, upper bound: 9160.7145579
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.7117217
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.7108863, upper bound: 9160.7145579
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9160.5848627, upper bound: 9160.7117217
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.6707649, upper bound: 9159.7180838
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.7169340, upper bound: 9159.7169341
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.6707649, upper bound: 9160.5794123
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9159.7169340, upper bound: 9160.5782626
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 0, lower bound: -9158.5609829, upper bound: 9158.5578086
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 0, lower bound: -9158.5609829, upper bound: 9158.5623870
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.32
Output dim: 0, lower bound: -9158.5610246, upper bound: 9158.5578091
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 0, lower bound: -9158.5610246, upper bound: 9158.9519562

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6112.4824219, 4541.9643555, -9881.4277344, 10085.8037109
1: -4247.0512695, 3818.8703613, -4859.1616211, 4366.1567383, -8613.2080078, 8678.0312500
2: -6172.7690430, 4157.7558594, -7058.1166992, 4746.1420898, -10918.9091797, 11215.8730469
3: -2321.1672363, 5922.5683594, -2653.8247070, 6779.5839844, -9100.7509766, 8576.3925781
4: -6855.8618164, 4119.5634766, -7841.5019531, 4708.2548828, -11564.1162109, 11961.0654297

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
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
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1576437, upper bound: 9161.1576437
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1576437, upper bound: 9163.2054951
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6171.0551758, 4584.4877930, -10641.4794922, 10671.5761719
1: -4814.2382812, 4326.6323242, -4905.5644531, 4407.1103516, -9221.3447266, 9232.1962891
2: -6991.6362305, 4702.5898438, -7125.1625977, 4789.8227539, -11781.4589844, 11827.7509766
3: -2630.3791504, 6718.2563477, -2678.3530273, 6844.5351562, -9474.9140625, 9396.6093750
4: -7768.4980469, 4665.7036133, -7916.1811523, 4751.7998047, -12520.2958984, 12581.8837891

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2054951, upper bound: 9161.1612528
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2054951, upper bound: 9163.2091042
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6590.4291992, 4908.7729492, -10248.2363281, 10563.7509766
1: -4247.0512695, 3818.8703613, -5238.4340820, 4723.0971680, -8970.1455078, 9057.3027344
2: -6172.7690430, 4157.7558594, -7604.4345703, 5124.6596680, -11297.4257812, 11762.1904297
3: -2321.1672363, 5922.5683594, -2853.1391602, 7327.9360352, -9649.1035156, 8775.7070312
4: -6855.8618164, 4119.5634766, -8446.3496094, 5075.0302734, -11930.8916016, 12565.9130859

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
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
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.6907415
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.7209115
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6652.2324219, 4953.4042969, -11010.3964844, 11152.7539062
1: -4814.2382812, 4326.6323242, -5287.2490234, 4766.0849609, -9580.3222656, 9613.8808594
2: -6991.6362305, 4702.5898438, -7675.0820312, 5170.7749023, -12162.4111328, 12377.6708984
3: -2630.3791504, 6718.2563477, -2878.7878418, 7395.5224609, -10025.9013672, 9597.0439453
4: -7768.4980469, 4665.7036133, -8525.3037109, 5120.5253906, -12889.0214844, 13191.0048828

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1978397, upper bound: 9160.6943506
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1978397, upper bound: 9160.7245129
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6112.4824219, 4541.9643555, -10390.4726562, 10470.1777344
1: -4652.5683594, 4191.5620117, -4859.1616211, 4366.1567383, -9018.7246094, 9050.7236328
2: -6754.8217773, 4550.6113281, -7058.1166992, 4746.1420898, -11500.9628906, 11608.7285156
3: -2527.9526367, 6502.3999023, -2653.8247070, 6779.5839844, -9307.5371094, 9156.2246094
4: -7498.1806641, 4504.6772461, -7841.5019531, 4708.2548828, -12206.4335938, 12346.1796875

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
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
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6907415, upper bound: 9161.1499883
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6907415, upper bound: 9163.1978397
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6171.0551758, 4584.4877930, -11119.3613281, 11039.9570312
1: -5194.1127930, 4684.6015625, -4905.5644531, 4407.1103516, -9601.2197266, 9590.1640625
2: -7539.5957031, 5082.8535156, -7125.1625977, 4789.8227539, -12329.4179688, 12208.0156250
3: -2830.4941406, 7265.7309570, -2678.3530273, 6844.5351562, -9675.0292969, 9944.0820312
4: -8374.8789062, 5033.6191406, -7916.1811523, 4751.7998047, -13126.6777344, 12949.8007812

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7209115, upper bound: 9161.1499883
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7209115, upper bound: 9163.1978397
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6575.6201172, 4900.3725586, -10748.8828125, 10933.3154297
1: -4652.5683594, 4191.5620117, -5226.9838867, 4714.8764648, -9367.4414062, 9418.5449219
2: -6754.8217773, 4550.6113281, -7588.4467773, 5116.0971680, -11870.9179688, 12139.0585938
3: -2527.9526367, 6502.3999023, -2848.9687500, 7313.7495117, -9841.7021484, 9351.3691406
4: -7498.1806641, 4504.6772461, -8428.3349609, 5066.7734375, -12564.9541016, 12933.0117188

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
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
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830862, upper bound: 9160.6830862
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830862, upper bound: 9160.7132561
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6636.2197266, 4944.3427734, -11479.2148438, 11505.1201172
1: -5194.1127930, 4684.6015625, -5274.8398438, 4757.2133789, -9951.3242188, 9959.4414062
2: -7539.5957031, 5082.8535156, -7657.7958984, 5161.5336914, -12701.1289062, 12740.6494141
3: -2830.4941406, 7265.7309570, -2874.2885742, 7380.1958008, -10210.6894531, 10140.0166016
4: -8374.8789062, 5033.6191406, -8505.7568359, 5111.6137695, -13486.4921875, 13539.3759766

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.6830862
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.7132561
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8193.0019531, 6075.2744141, -11414.7373047, 12166.3212891
1: -4247.0512695, 3818.8703613, -6505.1440430, 5841.7416992, -10088.7919922, 10324.0126953
2: -6172.7690430, 4157.7558594, -9433.9218750, 6371.9707031, -12544.7392578, 13591.6777344
3: -2321.1672363, 5922.5683594, -3586.6989746, 9066.8535156, -11388.0205078, 9509.2675781
4: -6855.8618164, 4119.5634766, -10477.8701172, 6321.7441406, -13177.6054688, 14597.4335938

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1433882, upper bound: 9159.6850203
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1433882, upper bound: 9159.7311895
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8260.9833984, 6124.1367188, -12181.1289062, 12761.5039062
1: -4814.2382812, 4326.6323242, -6558.9648438, 5888.8247070, -10703.0615234, 10885.5976562
2: -6991.6362305, 4702.5898438, -9511.6953125, 6422.5688477, -13414.2041016, 14214.2841797
3: -2630.3791504, 6718.2563477, -3614.8420410, 9141.4384766, -11771.8173828, 10333.0986328
4: -7768.4980469, 4665.7036133, -10564.5058594, 6371.6967773, -14140.1933594, 15230.2070312

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1912396, upper bound: 9159.6886099
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1912396, upper bound: 9159.7346930
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8838.8154297, 6552.2543945, -11891.7177734, 12812.1357422
1: -4247.0512695, 3818.8703613, -7015.7568359, 6302.1748047, -10549.2255859, 10834.6259766
2: -6172.7690430, 4157.7558594, -10169.2236328, 6866.2788086, -13039.0478516, 14326.9794922
3: -2321.1672363, 5922.5683594, -3844.1459961, 9786.6865234, -12107.8535156, 9766.7138672
4: -6855.8618164, 4119.5634766, -11288.2685547, 6803.5249023, -13659.3847656, 15407.8310547

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1484539, upper bound: 9160.5925179
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1484539, upper bound: 9160.5925179
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8904.5888672, 6599.2202148, -12656.2119141, 13405.1093750
1: -4814.2382812, 4326.6323242, -7067.9404297, 6347.4223633, -11161.6601562, 11394.5722656
2: -6991.6362305, 4702.5898438, -10244.5087891, 6914.9130859, -13906.5488281, 14947.0976562
3: -2630.3791504, 6718.2563477, -3871.0993652, 9858.5449219, -12488.9238281, 10589.3554688
4: -7768.4980469, 4665.7036133, -11372.1064453, 6851.4018555, -14619.8994141, 16037.8085938

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1963053, upper bound: 9160.5960991
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1963053, upper bound: 9160.5960991
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8193.0019531, 6075.2744141, -11923.7822266, 12550.6972656
1: -4652.5683594, 4191.5620117, -6505.1440430, 5841.7416992, -10494.3076172, 10696.7050781
2: -6754.8217773, 4550.6113281, -9433.9218750, 6371.9707031, -13126.7929688, 13984.5332031
3: -2527.9526367, 6502.3999023, -3586.6989746, 9066.8535156, -11594.8066406, 10089.0986328
4: -7498.1806641, 4504.6772461, -10477.8701172, 6321.7441406, -13819.9248047, 14982.5468750

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6764861, upper bound: 9159.6773650
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6764861, upper bound: 9159.7235342
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8260.9833984, 6124.1367188, -12659.0097656, 13129.8847656
1: -5194.1127930, 4684.6015625, -6558.9648438, 5888.8247070, -11082.9355469, 11243.5654297
2: -7539.5957031, 5082.8535156, -9511.6953125, 6422.5688477, -13962.1640625, 14594.5488281
3: -2830.4941406, 7265.7309570, -3614.8420410, 9141.4384766, -11971.9326172, 10880.5712891
4: -8374.8789062, 5033.6191406, -10564.5058594, 6371.6967773, -14746.5751953, 15598.1240234

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7066560, upper bound: 9159.6773650
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7066560, upper bound: 9159.7235342
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8838.8154297, 6552.2543945, -12400.7626953, 13195.8320312
1: -4652.5683594, 4191.5620117, -7015.7568359, 6302.1748047, -10954.7431641, 11207.3183594
2: -6754.8217773, 4550.6113281, -10169.2236328, 6866.2788086, -13621.1005859, 14719.8349609
3: -2527.9526367, 6502.3999023, -3844.1459961, 9786.6865234, -12314.6386719, 10346.5458984
4: -7498.1806641, 4504.6772461, -11288.2685547, 6803.5249023, -14301.7031250, 15792.9453125

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6764861, upper bound: 9160.5840082
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6764861, upper bound: 9160.5840082
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8904.5888672, 6599.2202148, -13134.0908203, 13773.4892578
1: -5194.1127930, 4684.6015625, -7067.9404297, 6347.4223633, -11541.5341797, 11752.5410156
2: -7539.5957031, 5082.8535156, -10244.5087891, 6914.9130859, -14454.5087891, 15327.3623047
3: -2830.4941406, 7265.7309570, -3871.0993652, 9858.5449219, -12689.0390625, 11136.8281250
4: -8374.8789062, 5033.6191406, -11372.1064453, 6851.4018555, -15226.2812500, 16405.7265625

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7066560, upper bound: 9160.5840082
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7066560, upper bound: 9160.5840082
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -6112.4824219, 4541.9643555, -11951.5429688, 11619.8496094
1: -5884.5581055, 5296.7612305, -4859.1616211, 4366.1567383, -10250.7148438, 10155.9228516
2: -8539.4326172, 5781.5205078, -7058.1166992, 4746.1420898, -13285.5742188, 12839.6367188
3: -3248.6958008, 8209.8476562, -2653.8247070, 6779.5839844, -10028.2792969, 10863.6718750
4: -9481.6269531, 5730.7734375, -7841.5019531, 4708.2548828, -14189.8808594, 13572.2753906

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
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
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6850203, upper bound: 9161.1433882
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6850203, upper bound: 9163.1912396
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -6171.0551758, 4584.4877930, -12754.9804688, 12229.7919922
1: -6487.3271484, 5825.9394531, -4905.5644531, 4407.1103516, -10894.4355469, 10731.5029297
2: -9407.1347656, 6354.2421875, -7125.1625977, 4789.8227539, -14196.9570312, 13479.4023438
3: -3577.6472168, 9041.6318359, -2678.3530273, 6844.5351562, -10422.1826172, 11719.9843750
4: -10448.0498047, 6304.1855469, -7916.1811523, 4751.7998047, -15199.8476562, 14220.3671875

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7311895, upper bound: 9161.1433882
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7311895, upper bound: 9163.1912396
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -6590.4291992, 4908.7729492, -12318.3505859, 12097.7958984
1: -5884.5581055, 5296.7612305, -5238.4340820, 4723.0971680, -10607.6533203, 10535.1943359
2: -8539.4326172, 5781.5205078, -7604.4345703, 5124.6596680, -13664.0917969, 13385.9550781
3: -3248.6958008, 8209.8476562, -2853.1391602, 7327.9360352, -10576.6318359, 11062.9863281
4: -9481.6269531, 5730.7734375, -8446.3496094, 5075.0302734, -14556.6572266, 14177.1220703

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
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
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.6764861
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.7066560
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -6652.2324219, 4953.4042969, -13123.8964844, 12710.9697266
1: -6487.3271484, 5825.9394531, -5287.2490234, 4766.0849609, -11253.4121094, 11113.1884766
2: -9407.1347656, 6354.2421875, -7675.0820312, 5170.7749023, -14577.9091797, 14029.3222656
3: -3577.6472168, 9041.6318359, -2878.7878418, 7395.5224609, -10973.1679688, 11920.4189453
4: -10448.0498047, 6304.1855469, -8525.3037109, 5120.5253906, -15568.5751953, 14829.4892578

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7235342, upper bound: 9160.6764861
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7235342, upper bound: 9160.7066560
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8139.2573242, 6046.2695312, -6112.4824219, 4541.9643555, -12681.2216797, 12158.7509766
1: -6461.4291992, 5817.6831055, -4859.1616211, 4366.1567383, -10827.5859375, 10676.8447266
2: -9369.9003906, 6341.2602539, -7058.1166992, 4746.1420898, -14116.0419922, 13399.3769531
3: -3544.0375977, 9022.6132812, -2653.8247070, 6779.5839844, -10323.6210938, 11676.4375000
4: -10397.9277344, 6278.9721680, -7841.5019531, 4708.2548828, -15106.1826172, 14120.4746094

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
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
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5925179, upper bound: 9161.1484539
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5925179, upper bound: 9163.1963053
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8814.8632812, 6533.5449219, -6171.0551758, 4584.4877930, -13399.3515625, 12704.5996094
1: -6996.9555664, 6284.2514648, -4905.5644531, 4407.1103516, -11404.0634766, 11189.8144531
2: -10141.1152344, 6846.4956055, -7125.1625977, 4789.8227539, -14930.9375000, 13971.6572266
3: -3833.6420898, 9759.3496094, -2678.3530273, 6844.5351562, -10678.1777344, 12437.7031250
4: -11256.9599609, 6783.5244141, -7916.1811523, 4751.7998047, -16008.7578125, 14699.7041016

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5925179, upper bound: 9161.1484539
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5925179, upper bound: 9163.1963053
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8139.2573242, 6046.2695312, -6575.6201172, 4900.3725586, -13039.6298828, 12621.8867188
1: -6461.4291992, 5817.6831055, -5226.9838867, 4714.8764648, -11176.3037109, 11044.6660156
2: -9369.9003906, 6341.2602539, -7588.4467773, 5116.0971680, -14485.9970703, 13929.7070312
3: -3544.0375977, 9022.6132812, -2848.9687500, 7313.7495117, -10857.7871094, 11871.5820312
4: -10397.9277344, 6278.9721680, -8428.3349609, 5066.7734375, -15464.7011719, 14707.3066406

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
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
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.1509623, upper bound: 9157.5704517
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.6815518
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.7117217
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8814.8632812, 6533.5449219, -6636.2197266, 4944.3427734, -13759.2050781, 13169.7626953
1: -6996.9555664, 6284.2514648, -5274.8398438, 4757.2133789, -11754.1689453, 11559.0908203
2: -10141.1152344, 6846.4956055, -7657.7958984, 5161.5336914, -15302.6484375, 14504.2910156
3: -3833.6420898, 9759.3496094, -2874.2885742, 7380.1958008, -11213.8369141, 12633.6367188
4: -11256.9599609, 6783.5244141, -8505.7568359, 5111.6137695, -16368.5742188, 15289.2802734

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3377260, upper bound: 9157.5641233
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848627, upper bound: 9160.6815518
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.7117217
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -8193.0019531, 6075.2744141, -13484.8515625, 13699.1904297
1: -5884.5581055, 5296.7612305, -6505.1440430, 5841.7416992, -11726.2978516, 11801.9042969
2: -8539.4326172, 5781.5205078, -9433.9218750, 6371.9707031, -14911.4033203, 15212.9785156
3: -3248.6958008, 8209.8476562, -3586.6989746, 9066.8535156, -12309.9189453, 11792.2500000
4: -9481.6269531, 5730.7734375, -10477.8701172, 6321.7441406, -15799.3085938, 16200.8281250

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
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

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6707649, upper bound: 9159.6707649
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6707649, upper bound: 9159.7169341
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -8260.9833984, 6124.1367188, -14294.6289062, 14319.7197266
1: -6487.3271484, 5825.9394531, -6558.9648438, 5888.8247070, -12376.1494141, 12384.9042969
2: -9407.1347656, 6354.2421875, -9511.6953125, 6422.5688477, -15829.7031250, 15864.8554688
3: -3577.6472168, 9041.6318359, -3614.8420410, 9141.4384766, -12714.8037109, 12652.8398438
4: -10448.0498047, 6304.1855469, -10564.5058594, 6371.6967773, -16816.0058594, 16863.1328125

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7169341, upper bound: 9159.6707649
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7169341, upper bound: 9159.7169341
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -8838.8154297, 6552.2543945, -13958.6728516, 14337.1699219
1: -5884.5581055, 5296.7612305, -7015.7568359, 6302.1748047, -12186.7324219, 12309.6152344
2: -8539.4326172, 5781.5205078, -10169.2236328, 6866.2788086, -15402.5234375, 15938.1347656
3: -3248.6958008, 8209.8476562, -3844.1459961, 9786.6865234, -13013.3847656, 12053.9921875
4: -9481.6269531, 5730.7734375, -11288.2685547, 6803.5249023, -16278.1757812, 16999.9804688

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6758306, upper bound: 9160.5782625
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6758306, upper bound: 9160.5782625
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -8904.5888672, 6599.2202148, -14767.5791016, 14955.5419922
1: -6487.3271484, 5825.9394531, -7067.9404297, 6347.4223633, -12834.7480469, 12892.8056641
2: -9407.1347656, 6354.2421875, -10244.5087891, 6914.9130859, -16319.1035156, 16587.4355469
3: -3577.6472168, 9041.6318359, -3871.0993652, 9858.5449219, -13415.5429688, 12912.7314453
4: -10448.0498047, 6304.1855469, -11372.1064453, 6851.4018555, -17292.9101562, 17659.3769531

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7219997, upper bound: 9160.5782625
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7219997, upper bound: 9160.5782625
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10558.6601562, 7805.8593750, -9839.7382812, 7272.6079102, -17719.4707031, 17546.6796875
1: -8370.0732422, 7519.9335938, -7801.6376953, 7000.6318359, -15300.7519531, 15259.0761719
2: -12145.7324219, 8180.4584961, -11326.9697266, 7627.3330078, -19639.8457031, 19390.9941406
3: -4551.9360352, 11683.6074219, -4267.9921875, 10881.2626953, -15370.7695312, 15862.0185547
4: -13489.3945312, 8092.2968750, -12583.8447266, 7551.2636719, -20884.7324219, 20536.5917969

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.3698619, upper bound: 9157.3742628
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.3696695, upper bound: 9158.6755797
time: 0.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.54 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1576437, upper bound: 9161.1576437
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1576437, upper bound: 9163.2054951
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.2054951, upper bound: 9161.1612528
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.2054951, upper bound: 9163.2091042
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.6907415
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1499883, upper bound: 9160.7209115
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.1978397, upper bound: 9160.6943506
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.1978397, upper bound: 9160.7245129
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6907415, upper bound: 9161.1499883
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6907415, upper bound: 9163.1978397
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7209115, upper bound: 9161.1499883
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7209115, upper bound: 9163.1978397
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6830862, upper bound: 9160.6830862
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6830862, upper bound: 9160.7132561
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.6830862
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7132561, upper bound: 9160.7132561
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1433882, upper bound: 9159.6850203
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1433882, upper bound: 9159.7311895
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.1912396, upper bound: 9159.6886099
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.1912396, upper bound: 9159.7346930
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1484539, upper bound: 9160.5925179
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9161.1484539, upper bound: 9160.5925179
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.1963053, upper bound: 9160.5960991
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9163.1963053, upper bound: 9160.5960991
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6764861, upper bound: 9159.6773650
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6764861, upper bound: 9159.7235342
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7066560, upper bound: 9159.6773650
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7066560, upper bound: 9159.7235342
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6764861, upper bound: 9160.5840082
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.6764861, upper bound: 9160.5840082
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7066560, upper bound: 9160.5840082
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.7066560, upper bound: 9160.5840082
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6850203, upper bound: 9161.1433882
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6850203, upper bound: 9163.1912396
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7311895, upper bound: 9161.1433882
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7311895, upper bound: 9163.1912396
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.6764861
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6773650, upper bound: 9160.7066560
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7235342, upper bound: 9160.6764861
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7235342, upper bound: 9160.7066560
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5925179, upper bound: 9161.1484539
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5925179, upper bound: 9163.1963053
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5925179, upper bound: 9161.1484539
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5925179, upper bound: 9163.1963053
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.6815518
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.7117217
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5848627, upper bound: 9160.6815518
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9160.5848626, upper bound: 9160.7117217
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6707649, upper bound: 9159.6707649
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6707649, upper bound: 9159.7169341
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7169341, upper bound: 9159.6707649
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7169341, upper bound: 9159.7169341
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6758306, upper bound: 9160.5782625
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.6758306, upper bound: 9160.5782625
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7219997, upper bound: 9160.5782625
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 0, lower bound: -9159.7219997, upper bound: 9160.5782625
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.54
Output dim: 0, lower bound: -9158.3698619, upper bound: 9157.3742628
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.54
Output dim: 0, lower bound: -9158.3696695, upper bound: 9158.6755797

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -5339.4643555, 3973.3225098, -9312.7841797, 9312.7841797
1: -4247.0512695, 3818.8703613, -4247.0512695, 3818.8703613, -8065.9208984, 8065.9204102
2: -6172.7690430, 4157.7558594, -6172.7690430, 4157.7558594, -10330.5244141, 10330.5253906
3: -2321.1672363, 5922.5683594, -2321.1672363, 5922.5683594, -8243.7353516, 8243.7353516
4: -6855.8618164, 4119.5634766, -6855.8618164, 4119.5634766, -10975.4248047, 10975.4248047

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1732210, upper bound: 9160.7297273
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7291016, upper bound: 9160.7291022
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6056.9916992, 4500.5209961, -9839.9853516, 10030.3134766
1: -4247.0512695, 3818.8703613, -4814.2382812, 4326.6323242, -8573.6835938, 8633.1074219
2: -6172.7690430, 4157.7558594, -6991.6362305, 4702.5898438, -10875.3574219, 11149.3925781
3: -2321.1672363, 5922.5683594, -2630.3791504, 6718.2563477, -9039.4238281, 8552.9472656
4: -6855.8618164, 4119.5634766, -7768.4980469, 4665.7036133, -11521.5634766, 11888.0605469

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1732210, upper bound: 9162.9424189
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7291016, upper bound: 9162.9417938
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -5339.4643555, 3973.3225098, -10030.3134766, 9839.9853516
1: -4814.2382812, 4326.6323242, -4247.0512695, 3818.8703613, -8633.1064453, 8573.6835938
2: -6991.6362305, 4702.5898438, -6172.7690430, 4157.7558594, -11149.3925781, 10875.3564453
3: -2630.3791504, 6718.2563477, -2321.1672363, 5922.5683594, -8552.9472656, 9039.4238281
4: -7768.4980469, 4665.7036133, -6855.8618164, 4119.5634766, -11888.0605469, 11521.5625000

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1822313, upper bound: 9155.6202835
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1973388, upper bound: 9155.6214136
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6056.9916992, 4500.5209961, -10557.5126953, 10557.5126953
1: -4814.2382812, 4326.6323242, -4814.2382812, 4326.6323242, -9140.8691406, 9140.8701172
2: -6991.6362305, 4702.5898438, -6991.6362305, 4702.5898438, -11694.2265625, 11694.2255859
3: -2630.3791504, 6718.2563477, -2630.3791504, 6718.2563477, -9348.6357422, 9348.6357422
4: -7768.4980469, 4665.7036133, -7768.4980469, 4665.7036133, -12434.2001953, 12434.1992188

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1822316, upper bound: 9163.2054489
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1973391, upper bound: 9163.2084311
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -5848.5097656, 4357.6953125, -9697.1601562, 9821.8300781
1: -4247.0512695, 3818.8703613, -4652.5683594, 4191.5620117, -8438.6113281, 8471.4365234
2: -6172.7690430, 4157.7558594, -6754.8217773, 4550.6113281, -10723.3798828, 10912.5781250
3: -2321.1672363, 5922.5683594, -2527.9526367, 6502.3999023, -8823.5673828, 8450.5214844
4: -6855.8618164, 4119.5634766, -7498.1806641, 4504.6772461, -11360.5390625, 11617.7431641

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1675249, upper bound: 9160.3900597
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7234055, upper bound: 9160.3894346
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6534.8730469, 4868.9013672, -10208.3652344, 10508.1943359
1: -4247.0512695, 3818.8703613, -5194.1127930, 4684.6015625, -8931.6513672, 9012.9804688
2: -6172.7690430, 4157.7558594, -7539.5957031, 5082.8535156, -11255.6220703, 11697.3515625
3: -2321.1672363, 5922.5683594, -2830.4941406, 7265.7309570, -9586.8964844, 8753.0625000
4: -6855.8618164, 4119.5634766, -8374.8789062, 5033.6191406, -11889.4804688, 12494.4423828

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1675249, upper bound: 9160.4536527
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7234055, upper bound: 9160.4530276
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -5848.5097656, 4357.6953125, -10414.6875000, 10349.0302734
1: -4814.2382812, 4326.6323242, -4652.5683594, 4191.5620117, -9005.7988281, 8979.2001953
2: -6991.6362305, 4702.5898438, -6754.8217773, 4550.6113281, -11542.2480469, 11457.4111328
3: -2630.3791504, 6718.2563477, -2527.9526367, 6502.3999023, -9132.7792969, 9246.2089844
4: -7768.4980469, 4665.7036133, -7498.1806641, 4504.6772461, -12273.1748047, 12163.8818359

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4457202, upper bound: 9160.3955630
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9360972, upper bound: 9160.3955438
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6534.8730469, 4868.9013672, -10925.8925781, 11035.3935547
1: -4814.2382812, 4326.6323242, -5194.1127930, 4684.6015625, -9498.8378906, 9520.7441406
2: -6991.6362305, 4702.5898438, -7539.5957031, 5082.8535156, -12074.4902344, 12242.1845703
3: -2630.3791504, 6718.2563477, -2830.4941406, 7265.7309570, -9896.1103516, 9548.7500000
4: -7768.4980469, 4665.7036133, -8374.8789062, 5033.6191406, -12802.1171875, 13040.5820312

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4457209, upper bound: 9160.4588357
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9360975, upper bound: 9160.4590629
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -5339.4643555, 3973.3225098, -9821.8300781, 9697.1601562
1: -4652.5683594, 4191.5620117, -4247.0512695, 3818.8703613, -8471.4365234, 8438.6113281
2: -6754.8217773, 4550.6113281, -6172.7690430, 4157.7558594, -10912.5781250, 10723.3798828
3: -2527.9526367, 6502.3999023, -2321.1672363, 5922.5683594, -8450.5214844, 8823.5673828
4: -7498.1806641, 4504.6772461, -6855.8618164, 4119.5634766, -11617.7431641, 11360.5390625

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6623119, upper bound: 9160.7227947
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3895832, upper bound: 9160.7241026
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6056.9916992, 4500.5209961, -10349.0302734, 10414.6875000
1: -4652.5683594, 4191.5620117, -4814.2382812, 4326.6323242, -8979.2001953, 9005.7978516
2: -6754.8217773, 4550.6113281, -6991.6362305, 4702.5898438, -11457.4111328, 11542.2480469
3: -2527.9526367, 6502.3999023, -2630.3791504, 6718.2563477, -9246.2089844, 9132.7792969
4: -7498.1806641, 4504.6772461, -7768.4980469, 4665.7036133, -12163.8818359, 12273.1757812

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6623119, upper bound: 9162.9354863
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3895832, upper bound: 9162.9367942
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -5339.4643555, 3973.3225098, -10508.1933594, 10208.3652344
1: -5194.1127930, 4684.6015625, -4247.0512695, 3818.8703613, -9012.9794922, 8931.6513672
2: -7539.5957031, 5082.8535156, -6172.7690430, 4157.7558594, -11697.3515625, 11255.6210938
3: -2830.4941406, 7265.7309570, -2321.1672363, 5922.5683594, -8753.0625000, 9586.8964844
4: -8374.8789062, 5033.6191406, -6855.8618164, 4119.5634766, -12494.4423828, 11889.4804688

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5282012, upper bound: 9159.9554414
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5760262, upper bound: 9158.7795737
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2047994, upper bound: 9161.0973352
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6665065, upper bound: 9161.1016846
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6056.9916992, 4500.5209961, -11035.3935547, 10925.8925781
1: -5194.1127930, 4684.6015625, -4814.2382812, 4326.6323242, -9520.7441406, 9498.8388672
2: -7539.5957031, 5082.8535156, -6991.6362305, 4702.5898438, -12242.1855469, 12074.4902344
3: -2830.4941406, 7265.7309570, -2630.3791504, 6718.2563477, -9548.7500000, 9896.1093750
4: -8374.8789062, 5033.6191406, -7768.4980469, 4665.7036133, -13040.5820312, 12802.1171875

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5282012, upper bound: 9161.3519029
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5760263, upper bound: 9160.1799708
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2047994, upper bound: 9162.2454338
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6665065, upper bound: 9162.1422138
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -5848.5097656, 4357.6953125, -10206.2050781, 10206.2050781
1: -4652.5683594, 4191.5620117, -4652.5683594, 4191.5620117, -8844.1279297, 8844.1279297
2: -6754.8217773, 4550.6113281, -6754.8217773, 4550.6113281, -11305.4335938, 11305.4335938
3: -2527.9526367, 6502.3999023, -2527.9526367, 6502.3999023, -9030.3525391, 9030.3525391
4: -7498.1806641, 4504.6772461, -7498.1806641, 4504.6772461, -12002.8574219, 12002.8574219

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6566158, upper bound: 9160.3831272
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3838871, upper bound: 9160.3844350
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6521.0463867, 4861.1459961, -10709.6562500, 10878.7421875
1: -4652.5683594, 4191.5620117, -5183.4223633, 4677.0102539, -9329.5751953, 9374.9824219
2: -6754.8217773, 4550.6113281, -7524.6733398, 5074.9497070, -11829.7714844, 12075.2851562
3: -2527.9526367, 6502.3999023, -2826.6457520, 7252.5444336, -9780.4970703, 9329.0458984
4: -7498.1806641, 4504.6772461, -8358.0966797, 5025.9990234, -12524.1787109, 12862.7734375

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6566158, upper bound: 9160.4467201
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3838871, upper bound: 9160.4480280
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -5848.5097656, 4357.6953125, -10892.5683594, 10717.4091797
1: -5194.1127930, 4684.6015625, -4652.5683594, 4191.5620117, -9385.6718750, 9337.1679688
2: -7539.5957031, 5082.8535156, -6754.8217773, 4550.6113281, -12090.2070312, 11837.6757812
3: -2830.4941406, 7265.7309570, -2527.9526367, 6502.3999023, -9332.8945312, 9793.6816406
4: -8374.8789062, 5033.6191406, -7498.1806641, 4504.6772461, -12879.5566406, 12531.7998047

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1971440, upper bound: 9160.6545018
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6588511, upper bound: 9160.6588512
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6521.0463867, 4861.1459961, -11396.0195312, 11389.9472656
1: -5194.1127930, 4684.6015625, -5183.4223633, 4677.0102539, -9871.1191406, 9868.0224609
2: -7539.5957031, 5082.8535156, -7524.6733398, 5074.9497070, -12614.5449219, 12607.5263672
3: -2830.4941406, 7265.7309570, -2826.6457520, 7252.5444336, -10083.0390625, 10092.3759766
4: -8374.8789062, 5033.6191406, -8358.0966797, 5025.9990234, -13400.8779297, 13391.7158203

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1971440, upper bound: 9160.6545018
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6588511, upper bound: 9160.6588512
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -7409.5781250, 5507.3666992, -10846.8310547, 11382.8994141
1: -4247.0512695, 3818.8703613, -5884.5581055, 5296.7612305, -9543.8115234, 9703.4267578
2: -6172.7690430, 4157.7558594, -8539.4326172, 5781.5205078, -11954.2880859, 12697.1884766
3: -2321.1672363, 5922.5683594, -3248.6958008, 8209.8476562, -10531.0146484, 9171.2636719
4: -6855.8618164, 4119.5634766, -9481.6269531, 5730.7734375, -12586.6328125, 13601.1904297

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1614592, upper bound: 9159.4219371
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7173399, upper bound: 9159.4213120
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8170.4921875, 6058.7373047, -11398.2001953, 12143.8134766
1: -4247.0512695, 3818.8703613, -6487.3271484, 5825.9394531, -10072.9902344, 10306.1943359
2: -6172.7690430, 4157.7558594, -9407.1347656, 6354.2421875, -12527.0097656, 13564.8906250
3: -2321.1672363, 5922.5683594, -3577.6472168, 9041.6318359, -11362.7988281, 9500.2158203
4: -6855.8618164, 4119.5634766, -10448.0498047, 6304.1855469, -13160.0468750, 14567.6123047

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1614592, upper bound: 9159.6085354
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7173399, upper bound: 9159.6079103
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -7409.5781250, 5507.3666992, -11564.3583984, 11910.0996094
1: -4814.2382812, 4326.6323242, -5884.5581055, 5296.7612305, -10110.9980469, 10211.1904297
2: -6991.6362305, 4702.5898438, -8539.4326172, 5781.5205078, -12773.1562500, 13242.0224609
3: -2630.3791504, 6718.2563477, -3248.6958008, 8209.8476562, -10840.2265625, 9966.9521484
4: -7768.4980469, 4665.7036133, -9481.6269531, 5730.7734375, -13499.2695312, 14147.3300781

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4396545, upper bound: 9159.4272294
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9300315, upper bound: 9159.4273721
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8170.4921875, 6058.7373047, -12115.7285156, 12671.0126953
1: -4814.2382812, 4326.6323242, -6487.3271484, 5825.9394531, -10640.1777344, 10813.9580078
2: -6991.6362305, 4702.5898438, -9407.1347656, 6354.2421875, -13345.8769531, 14109.7246094
3: -2630.3791504, 6718.2563477, -3577.6472168, 9041.6318359, -11672.0107422, 10295.9033203
4: -7768.4980469, 4665.7036133, -10448.0498047, 6304.1855469, -14072.6835938, 15113.7519531

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4396552, upper bound: 9159.6131566
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9300318, upper bound: 9159.6138712
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8139.2573242, 6046.2695312, -11385.7324219, 12112.5791016
1: -4247.0512695, 3818.8703613, -6461.4291992, 5817.6831055, -10064.7324219, 10280.2978516
2: -6172.7690430, 4157.7558594, -9369.9003906, 6341.2602539, -12514.0263672, 13527.6562500
3: -2321.1672363, 5922.5683594, -3544.0375977, 9022.6132812, -11343.7802734, 9466.6044922
4: -6855.8618164, 4119.5634766, -10397.9277344, 6278.9721680, -13134.8339844, 14517.4912109

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1664850, upper bound: 9160.3213564
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7223657, upper bound: 9160.3207313
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8814.8632812, 6533.5449219, -11873.0078125, 12788.1845703
1: -4247.0512695, 3818.8703613, -6996.9555664, 6284.2514648, -10531.3017578, 10815.8242188
2: -6172.7690430, 4157.7558594, -10141.1152344, 6846.4956055, -13019.2626953, 14298.8710938
3: -2321.1672363, 5922.5683594, -3833.6420898, 9759.3496094, -12080.5166016, 9756.2109375
4: -6855.8618164, 4119.5634766, -11256.9599609, 6783.5244141, -13639.3837891, 15376.5224609

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1664850, upper bound: 9160.3351155
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7223657, upper bound: 9160.3344904
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8139.2573242, 6046.2695312, -12103.2597656, 12639.7783203
1: -4814.2382812, 4326.6323242, -6461.4291992, 5817.6831055, -10631.9189453, 10788.0615234
2: -6991.6362305, 4702.5898438, -9369.9003906, 6341.2602539, -13332.8964844, 14072.4902344
3: -2630.3791504, 6718.2563477, -3544.0375977, 9022.6132812, -11652.9921875, 10262.2929688
4: -7768.4980469, 4665.7036133, -10397.9277344, 6278.9721680, -14047.4697266, 15063.6289062

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3833935, upper bound: 9157.3509390
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4446803, upper bound: 9160.3265328
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9350573, upper bound: 9160.3267652
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8814.8632812, 6533.5449219, -12590.5361328, 13315.3847656
1: -4814.2382812, 4326.6323242, -6996.9555664, 6284.2514648, -11098.4882812, 11323.5878906
2: -6991.6362305, 4702.5898438, -10141.1152344, 6846.4956055, -13838.1318359, 14843.7050781
3: -2630.3791504, 6718.2563477, -3833.6420898, 9759.3496094, -12389.7285156, 10551.8984375
4: -7768.4980469, 4665.7036133, -11256.9599609, 6783.5244141, -14552.0195312, 15922.6611328

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3833939, upper bound: 9157.3509390
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4446810, upper bound: 9160.3400831
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9350576, upper bound: 9160.3404963
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -7409.5781250, 5507.3666992, -11355.8750000, 11767.2734375
1: -4652.5683594, 4191.5620117, -5884.5581055, 5296.7612305, -9949.3281250, 10076.1181641
2: -6754.8217773, 4550.6113281, -8539.4326172, 5781.5205078, -12536.3408203, 13090.0439453
3: -2527.9526367, 6502.3999023, -3248.6958008, 8209.8476562, -10737.8007812, 9751.0957031
4: -7498.1806641, 4504.6772461, -9481.6269531, 5730.7734375, -13228.9511719, 13986.3046875

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6505501, upper bound: 9159.4150046
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3778215, upper bound: 9159.4163124
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8170.4921875, 6058.7373047, -11907.2451172, 12528.1875000
1: -4652.5683594, 4191.5620117, -6487.3271484, 5825.9394531, -10478.5078125, 10678.8867188
2: -6754.8217773, 4550.6113281, -9407.1347656, 6354.2421875, -13109.0615234, 13957.7460938
3: -2527.9526367, 6502.3999023, -3577.6472168, 9041.6318359, -11569.5839844, 10080.0468750
4: -7498.1806641, 4504.6772461, -10448.0498047, 6304.1855469, -13802.3662109, 14952.7265625

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6505501, upper bound: 9159.6016028
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3778215, upper bound: 9159.6029107
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -7409.5781250, 5507.3666992, -12042.2402344, 12278.4794922
1: -5194.1127930, 4684.6015625, -5884.5581055, 5296.7612305, -10490.8720703, 10569.1582031
2: -7539.5957031, 5082.8535156, -8539.4326172, 5781.5205078, -13321.1162109, 13622.2861328
3: -2830.4941406, 7265.7309570, -3248.6958008, 8209.8476562, -11040.3417969, 10514.4257812
4: -8374.8789062, 5033.6191406, -9481.6269531, 5730.7734375, -14105.6513672, 14515.2460938

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1905439, upper bound: 9159.5556015
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6522510, upper bound: 9159.5599510
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8170.4921875, 6058.7373047, -12593.6083984, 13039.3935547
1: -5194.1127930, 4684.6015625, -6487.3271484, 5825.9394531, -11020.0517578, 11171.9257812
2: -7539.5957031, 5082.8535156, -9407.1347656, 6354.2421875, -13893.8369141, 14489.9882812
3: -2830.4941406, 7265.7309570, -3577.6472168, 9041.6318359, -11872.1259766, 10843.3750000
4: -8374.8789062, 5033.6191406, -10448.0498047, 6304.1855469, -14679.0644531, 15481.6689453

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1905439, upper bound: 9159.5709311
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6522510, upper bound: 9159.5733227
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8139.2573242, 6046.2695312, -11894.7763672, 12496.9531250
1: -4652.5683594, 4191.5620117, -6461.4291992, 5817.6831055, -10470.2490234, 10652.9912109
2: -6754.8217773, 4550.6113281, -9369.9003906, 6341.2602539, -13096.0810547, 13920.5117188
3: -2527.9526367, 6502.3999023, -3544.0375977, 9022.6132812, -11550.5664062, 10046.4375000
4: -7498.1806641, 4504.6772461, -10397.9277344, 6278.9721680, -13777.1523438, 14902.6054688

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.6298005, upper bound: 9157.3240213
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6505501, upper bound: 9160.3136896
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3778215, upper bound: 9160.3150295
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8814.8632812, 6533.5449219, -12382.0527344, 13170.5761719
1: -4652.5683594, 4191.5620117, -6996.9555664, 6284.2514648, -10936.8183594, 11188.5166016
2: -6754.8217773, 4550.6113281, -10141.1152344, 6846.4956055, -13601.3173828, 14691.7265625
3: -2527.9526367, 6502.3999023, -3833.6420898, 9759.3496094, -12287.3027344, 10336.0419922
4: -7498.1806641, 4504.6772461, -11256.9599609, 6783.5244141, -14281.7021484, 15761.6367188

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.6298006, upper bound: 9157.3240213
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6505501, upper bound: 9160.3276066
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3778215, upper bound: 9160.3289260
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8139.2573242, 6046.2695312, -12581.1406250, 13008.1582031
1: -5194.1127930, 4684.6015625, -6461.4291992, 5817.6831055, -11011.7929688, 11146.0292969
2: -7539.5957031, 5082.8535156, -9369.9003906, 6341.2602539, -13880.8554688, 14452.7539062
3: -2830.4941406, 7265.7309570, -3544.0375977, 9022.6132812, -11853.1074219, 10809.7656250
4: -8374.8789062, 5033.6191406, -10397.9277344, 6278.9721680, -14653.8515625, 15431.5468750

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.5177298, upper bound: 9157.0803748
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1954086, upper bound: 9160.5002919
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6522510, upper bound: 9160.5034644
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8814.8632812, 6533.5449219, -13068.4169922, 13683.7646484
1: -5194.1127930, 4684.6015625, -6996.9555664, 6284.2514648, -11478.3623047, 11681.5556641
2: -7539.5957031, 5082.8535156, -10141.1152344, 6846.4956055, -14386.0917969, 15223.9687500
3: -2830.4941406, 7265.7309570, -3833.6420898, 9759.3496094, -12589.8437500, 11099.3710938
4: -8374.8789062, 5033.6191406, -11256.9599609, 6783.5244141, -15158.4013672, 16290.5791016

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.5177298, upper bound: 9157.0803748
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1954086, upper bound: 9160.5002919
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6522510, upper bound: 9160.5034644
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -5339.4643555, 3973.3225098, -11382.9003906, 10846.8310547
1: -5884.5581055, 5296.7612305, -4247.0512695, 3818.8703613, -9703.4267578, 9543.8115234
2: -8539.4326172, 5781.5205078, -6172.7690430, 4157.7558594, -12697.1884766, 11954.2861328
3: -3248.6958008, 8209.8476562, -2321.1672363, 5922.5683594, -9171.2636719, 10531.0146484
4: -9481.6269531, 5730.7734375, -6855.8618164, 4119.5634766, -13601.1904297, 12586.6337891

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.5367324, upper bound: 9159.9500005
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.5837973, upper bound: 9158.7740440
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3974029, upper bound: 9162.6371998
time: 0.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
time: 0.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 0, lower bound: -9164.3974029, upper bound: 9162.6371998
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -6543.1186523, 4870.7827148, -11168.8203125, 11224.5224609
1: -5006.5761719, 4500.5805664, -5201.3159180, 4682.8657227, -9689.4414062, 9701.8964844
2: -7271.7465820, 4893.0332031, -7552.2226562, 5097.3310547, -12369.0781250, 12445.2558594
3: -2736.1677246, 6988.3081055, -2850.7333984, 7262.3100586, -9998.4775391, 9839.0410156
4: -8079.3994141, 4853.4165039, -8390.7265625, 5055.5380859, -13134.9365234, 13244.1425781

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
time: 1.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
time: 0.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -6444.7890625, 4798.0073242, -13156.9091797, 12643.0400391
1: -6636.6801758, 5959.9243164, -5123.2690430, 4612.7319336, -11249.4121094, 11083.1933594
2: -9623.8066406, 6501.3457031, -7439.0156250, 5021.3212891, -14645.1279297, 13940.3613281
3: -3659.6928711, 9251.1083984, -2809.6154785, 7152.2675781, -10811.9609375, 12060.7236328
4: -10688.8964844, 6450.9218750, -8264.9394531, 4980.7612305, -15669.6582031, 14715.8603516

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
time: 0.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
time: 0.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -9162.6296732, upper bound: 9162.6296732

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -6298.0380859, 4681.4052734, -10979.4433594, 10979.4433594
1: -5006.5761719, 4500.5805664, -5006.5761719, 4500.5805664, -9507.1562500, 9507.1562500
2: -7271.7465820, 4893.0332031, -7271.7465820, 4893.0332031, -12164.7792969, 12164.7792969
3: -2736.1677246, 6988.3081055, -2736.1677246, 6988.3081055, -9724.4755859, 9724.4755859
4: -8079.3994141, 4853.4165039, -8079.3994141, 4853.4165039, -12932.8164062, 12932.8164062

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4392320, upper bound: 9161.5746609
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7299908, upper bound: 9161.5673311
time: 0.81 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -8358.9042969, 6198.2509766, -12496.2890625, 13040.3066406
1: -5006.5761719, 4500.5805664, -6636.6801758, 5959.9243164, -10966.5000000, 11137.2597656
2: -7271.7465820, 4893.0332031, -9623.8066406, 6501.3457031, -13773.0917969, 14516.8378906
3: -2736.1677246, 6988.3081055, -3659.6928711, 9251.1083984, -11987.2763672, 10648.0009766
4: -8079.3994141, 4853.4165039, -10688.8964844, 6450.9218750, -14530.3193359, 15542.3125000

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4392320, upper bound: 9161.5746609
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7299908, upper bound: 9161.5673311
time: 1.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -6298.0380859, 4681.4052734, -13040.3066406, 12496.2890625
1: -6636.6801758, 5959.9243164, -5006.5761719, 4500.5805664, -11137.2607422, 10966.5000000
2: -9623.8066406, 6501.3457031, -7271.7465820, 4893.0332031, -14516.8369141, 13773.0917969
3: -3659.6928711, 9251.1083984, -2736.1677246, 6988.3081055, -10648.0009766, 11987.2763672
4: -10688.8964844, 6450.9218750, -8079.3994141, 4853.4165039, -15542.3125000, 14530.3193359

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0963314, upper bound: 9161.5588059
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5666868, upper bound: 9161.5666872
time: 0.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -8358.9042969, 6198.2509766, -14557.1523438, 14557.1542969
1: -6636.6801758, 5959.9243164, -6636.6801758, 5959.9243164, -12596.6044922, 12596.6044922
2: -9623.8066406, 6501.3457031, -9623.8066406, 6501.3457031, -16125.1503906, 16125.1513672
3: -3659.6928711, 9251.1083984, -3659.6928711, 9251.1083984, -12910.8007812, 12910.8007812
4: -10688.8964844, 6450.9218750, -10688.8964844, 6450.9218750, -17137.2792969, 17137.2812500

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0963314, upper bound: 9161.5588059
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5666868, upper bound: 9161.5666872
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9163.4392320, upper bound: 9161.5746609
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9161.7299908, upper bound: 9161.5673311
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9163.4392320, upper bound: 9161.5746609
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9161.7299908, upper bound: 9161.5673311
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9160.0963314, upper bound: 9161.5588059
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9161.5666868, upper bound: 9161.5666872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9160.0963314, upper bound: 9161.5588059
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -9161.5666868, upper bound: 9161.5666872

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -6282.6020508, 4669.8134766, -10840.8691406, 10867.0898438
1: -4905.5644531, 4407.1103516, -4994.3105469, 4489.4125977, -9394.9746094, 9401.4199219
2: -7125.1625977, 4789.8227539, -7253.9760742, 4880.6992188, -12005.8613281, 12043.7988281
3: -2678.3530273, 6844.5351562, -2729.2849121, 6970.9755859, -9649.3281250, 9573.8203125
4: -7916.1811523, 4751.7998047, -8059.6166992, 4841.2680664, -12757.4492188, 12811.4160156

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1955296, upper bound: 9160.7066135
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0398819, upper bound: 9160.7204189
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -6186.7695312, 4596.9965820, -11249.2285156, 11140.1738281
1: -5287.2490234, 4766.0849609, -4918.0620117, 4419.9638672, -9707.2128906, 9684.1464844
2: -7675.0820312, 5170.7749023, -7143.4511719, 4804.4775391, -12479.5595703, 12314.2246094
3: -2878.7878418, 7395.5224609, -2685.7202148, 6863.3090820, -9742.0947266, 10081.2421875
4: -8525.3037109, 5120.5253906, -7936.6347656, 4765.3266602, -13290.6289062, 13057.1601562

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7148033, upper bound: 9160.6981230
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7131499, upper bound: 9160.7131498
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8346.8515625, 6189.1137695, -12360.1689453, 12931.3398438
1: -4905.5644531, 4407.1103516, -6627.1206055, 5951.1601562, -10856.7236328, 11034.2294922
2: -7125.1625977, 4789.8227539, -9610.0078125, 6491.6577148, -13616.8193359, 14399.8300781
3: -2678.3530273, 6844.5351562, -3654.2226562, 9237.6035156, -11915.9560547, 10498.7578125
4: -7916.1811523, 4751.7998047, -10673.5849609, 6441.2065430, -14357.3867188, 15425.3828125

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7220468, upper bound: 9160.0969962
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7220468, upper bound: 9161.5673311
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -8235.2402344, 6103.5712891, -12755.8037109, 13188.6445312
1: -5287.2490234, 4766.0849609, -6538.1733398, 5869.2724609, -11156.5214844, 11304.2568359
2: -7675.0820312, 5170.7749023, -9480.6796875, 6402.4589844, -14077.5410156, 14651.4531250
3: -2878.7878418, 7395.5224609, -3603.0727539, 9111.4658203, -11990.2529297, 10998.5957031
4: -8525.3037109, 5120.5253906, -10529.1113281, 6352.4243164, -14877.7285156, 15649.6357422

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7141865, upper bound: 9160.7108457
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7114963, upper bound: 9160.5848162
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -6282.6020508, 4669.8134766, -12930.7968750, 12406.7382812
1: -6558.9648438, 5888.8247070, -4994.3105469, 4489.4125977, -11048.3750000, 10883.1347656
2: -9511.6953125, 6422.5688477, -7253.9760742, 4880.6992188, -14392.3945312, 13676.5439453
3: -3614.8420410, 9141.4384766, -2729.2849121, 6970.9755859, -10585.8173828, 11870.7236328
4: -10564.5058594, 6371.6967773, -8059.6166992, 4841.2680664, -15405.7734375, 14431.3134766

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7248289, upper bound: 9160.6913185
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7233581, upper bound: 9160.7063119
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -6186.7695312, 4596.9965820, -13501.5830078, 12785.9902344
1: -7067.9404297, 6347.4223633, -4918.0620117, 4419.9638672, -11487.9042969, 11265.4843750
2: -10244.5087891, 6914.9130859, -7143.4511719, 4804.4775391, -15048.9863281, 14058.3642578
3: -3871.0993652, 9858.5449219, -2685.7202148, 6863.3090820, -10734.4062500, 12544.2656250
4: -11372.1064453, 6851.4018555, -7936.6347656, 4765.3266602, -16137.4316406, 14788.0371094

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5865492, upper bound: 9160.6964714
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.7114963
time: 2.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -8346.8515625, 6189.1137695, -14450.0976562, 14470.9882812
1: -6558.9648438, 5888.8247070, -6627.1206055, 5951.1601562, -12510.1250000, 12515.9433594
2: -9511.6953125, 6422.5688477, -9610.0078125, 6491.6577148, -16003.3525391, 16032.5761719
3: -3614.8420410, 9141.4384766, -3654.2226562, 9237.6035156, -12849.1230469, 12794.6767578
4: -10564.5058594, 6371.6967773, -10673.5849609, 6441.2065430, -17001.7636719, 17041.4003906

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0884045, upper bound: 9160.0884045
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0884045, upper bound: 9161.5588059
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8904.5888672, 6599.2202148, -8235.2402344, 6103.5712891, -15001.9482422, 14832.0673828
1: -7067.9404297, 6347.4223633, -6538.1733398, 5869.2724609, -12936.5722656, 12885.5947266
2: -10244.5087891, 6914.9130859, -9480.6796875, 6402.4589844, -16637.3886719, 16392.1816406
3: -3871.0993652, 9858.5449219, -3603.0727539, 9111.4658203, -12982.5644531, 13446.6630859
4: -11372.1064453, 6851.4018555, -10529.1113281, 6352.4243164, -17710.0292969, 17373.4453125

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.9507661, upper bound: 9158.6643161
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.9519448, upper bound: 9158.9512236
time: 0.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.87 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9163.1955296, upper bound: 9160.7066135
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9163.0398819, upper bound: 9160.7204189
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.7148033, upper bound: 9160.6981230
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.7131499, upper bound: 9160.7131498
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9161.7220468, upper bound: 9160.0969962
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9161.7220468, upper bound: 9161.5673311
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.7141865, upper bound: 9160.7108457
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.7114963, upper bound: 9160.5848162
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9159.7248289, upper bound: 9160.6913185
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9159.7233581, upper bound: 9160.7063119
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.5865492, upper bound: 9160.6964714
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.7114963
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.0884045, upper bound: 9160.0884045
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9160.0884045, upper bound: 9161.5588059
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9158.9507661, upper bound: 9158.6643161
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -9158.9519448, upper bound: 9158.9512236

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5893.0278320, 4381.7768555, -5445.6357422, 4052.3520508, -9945.3798828, 9827.4121094
1: -4685.4272461, 4211.9462891, -4331.4916992, 3895.1152344, -8580.5419922, 8543.4375000
2: -6806.8608398, 4581.1586914, -6295.1933594, 4241.0473633, -11047.9072266, 10876.3505859
3: -2561.5205078, 6536.8793945, -2367.3967285, 6042.9243164, -8604.4453125, 8904.2763672
4: -7561.6147461, 4543.7382812, -6992.0708008, 4201.9355469, -11763.5498047, 11535.8085938

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4267057, upper bound: 9160.3728688
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9323053, upper bound: 9160.3925932
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6140.0966797, 4562.1303711, -6159.1230469, 4579.4897461, -10719.5859375, 10721.2509766
1: -4880.9375000, 4385.7041016, -4895.5634766, 4402.8896484, -9283.8261719, 9281.2675781
2: -7089.0483398, 4766.5551758, -7109.7338867, 4786.8471680, -11875.8955078, 11876.2890625
3: -2665.4636230, 6810.9829102, -2677.7397461, 6834.8393555, -9500.2988281, 9488.7226562
4: -7875.9565430, 4728.7983398, -7899.9721680, 4748.5356445, -12624.4921875, 12628.7705078

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7188198
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7204189
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6356.8583984, 4739.3686523, -5347.4804688, 3977.3732910, -10334.2314453, 10086.8496094
1: -5054.8027344, 4559.6567383, -4253.3969727, 3823.1645508, -8877.9667969, 8813.0537109
2: -7337.8383789, 4948.9902344, -6182.0483398, 4161.4941406, -11499.3320312, 11131.0371094
3: -2755.4675293, 7071.1860352, -2321.5612793, 5932.7680664, -8688.2343750, 9392.7470703
4: -8148.1381836, 4901.7929688, -6866.1259766, 4122.8217773, -12270.9599609, 11767.9179688

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.6830114
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.6830114
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6619.8637695, 4930.4067383, -6053.0952148, 4500.6142578, -11120.4765625, 10983.5019531
1: -5261.4458008, 4743.9086914, -4811.1455078, 4327.5795898, -9589.0253906, 9555.0517578
2: -7637.7382812, 5146.8652344, -6987.8090820, 4704.2597656, -12341.9951172, 12134.6718750
3: -2865.7485352, 7359.8603516, -2630.7399902, 6716.7734375, -9582.5214844, 9990.5996094
4: -8484.0390625, 5096.9404297, -7764.5449219, 4666.1601562, -13150.1992188, 12861.4843750

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131499
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131499
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8260.9833984, 6124.1367188, -12295.1914062, 12845.4707031
1: -4905.5644531, 4407.1103516, -6558.9648438, 5888.8247070, -10794.3876953, 10966.0732422
2: -7125.1625977, 4789.8227539, -9511.6953125, 6422.5688477, -13547.7294922, 14301.5166016
3: -2678.3530273, 6844.5351562, -3614.8420410, 9141.4384766, -11819.7910156, 10459.3769531
4: -7916.1811523, 4751.7998047, -10564.5058594, 6371.6967773, -14287.8769531, 15316.3027344

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1429117, upper bound: 9159.7274302
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8201152, upper bound: 9159.7283536
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8904.5888672, 6599.2202148, -12770.2744141, 13489.0761719
1: -4905.5644531, 4407.1103516, -7067.9404297, 6347.4223633, -11252.9863281, 11475.0488281
2: -7125.1625977, 4789.8227539, -10244.5087891, 6914.9130859, -14040.0761719, 15034.3300781
3: -2678.3530273, 6844.5351562, -3871.0993652, 9858.5449219, -12536.8974609, 10715.6347656
4: -7916.1811523, 4751.7998047, -11372.1064453, 6851.4018555, -14767.5830078, 16123.9042969

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1429117, upper bound: 9160.5899267
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8201152, upper bound: 9160.5912281
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6356.8583984, 4739.3686523, -7388.3930664, 5490.2221680, -11847.0800781, 12127.7617188
1: -5054.8027344, 4559.6567383, -5867.1743164, 5280.5380859, -10335.3408203, 10426.8310547
2: -7337.8383789, 4948.9902344, -8513.6015625, 5764.6777344, -13102.5156250, 13462.5917969
3: -2755.4675293, 7071.1860352, -3238.8037109, 8184.8447266, -10940.3125000, 10309.9902344
4: -8148.1381836, 4901.7929688, -9452.1464844, 5714.6831055, -13862.8212891, 14353.9375000

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.5700515, upper bound: 9158.1509290
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6619.8637695, 4930.4067383, -8130.4882812, 6027.3403320, -12647.2031250, 13060.8945312
1: -5261.4458008, 4743.9086914, -6455.2001953, 5795.9819336, -11057.4277344, 11199.1093750
2: -7637.7382812, 5146.8652344, -9359.7275391, 6322.8823242, -13960.6201172, 14506.5927734
3: -2865.7485352, 7359.8603516, -3559.7922363, 8995.6894531, -11861.4375000, 10919.6523438
4: -8484.0390625, 5096.9404297, -10394.4736328, 6273.7998047, -14757.8388672, 15491.4121094

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.5639580, upper bound: 9157.3376537
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7932.6079102, 5888.1230469, -5445.6357422, 4052.3520508, -11984.9580078, 11333.7587891
1: -6298.9687500, 5661.4057617, -4331.4916992, 3895.1152344, -10194.0839844, 9992.8974609
2: -9136.0410156, 6177.9907227, -6295.1933594, 4241.0473633, -13377.0869141, 12473.1835938
3: -3478.8059082, 8781.1425781, -2367.3967285, 6042.9243164, -9521.7304688, 11148.5390625
4: -10146.0556641, 6130.1625977, -6992.0708008, 4201.9355469, -14347.9912109, 13122.2333984

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6269836, upper bound: 9159.2303453
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.6761740
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.6761740
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8239.5341797, 6108.5932617, -6159.1230469, 4579.4897461, -12819.0234375, 12267.7158203
1: -6541.9804688, 5873.8818359, -4895.5634766, 4402.8896484, -10944.8701172, 10769.4453125
2: -9486.9003906, 6406.3295898, -7109.7338867, 4786.8471680, -14273.7480469, 13516.0615234
3: -3605.9916992, 9117.7519531, -2677.7397461, 6834.8393555, -10440.8310547, 11795.4921875
4: -10536.8857422, 6355.6386719, -7899.9721680, 4748.5356445, -15285.4218750, 14255.6113281

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063120
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063120
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8594.0253906, 6377.8730469, -5347.4804688, 3977.3732910, -12571.3984375, 11725.3515625
1: -6821.5947266, 6134.5629883, -4253.3969727, 3823.1645508, -10644.7587891, 10387.9599609
2: -9889.2285156, 6685.9243164, -6182.0483398, 4161.4941406, -14050.7226562, 12867.9716797
3: -3743.8701172, 9519.6357422, -2321.5612793, 5932.7680664, -9676.6367188, 11841.1972656
4: -10976.7119141, 6625.5957031, -6866.1259766, 4122.8217773, -15099.5332031, 13491.7216797

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4125636, upper bound: 9159.2354285
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.6813391
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.6813391
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8876.2968750, 6578.6982422, -6053.0952148, 4500.6142578, -13376.9091797, 12631.7929688
1: -7045.5312500, 6327.6855469, -4811.1455078, 4327.5795898, -11373.1113281, 11138.8310547
2: -10211.9345703, 6893.5961914, -6987.8090820, 4704.2597656, -14916.1933594, 13881.4042969
3: -3859.4755859, 9827.4121094, -2630.7399902, 6716.7734375, -10576.2480469, 12458.1513672
4: -11335.8085938, 6830.2939453, -7764.5449219, 4666.1601562, -16001.9687500, 14594.8388672

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.7114963
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.7114963
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -8260.9833984, 6124.1367188, -14385.1201172, 14385.1201172
1: -6558.9648438, 5888.8247070, -6558.9648438, 5888.8247070, -12447.7890625, 12447.7880859
2: -9511.6953125, 6422.5688477, -9511.6953125, 6422.5688477, -15933.5830078, 15933.5820312
3: -3614.8420410, 9141.4384766, -3614.8420410, 9141.4384766, -12752.3291016, 12752.3291016
4: -10564.5058594, 6371.6967773, -10564.5058594, 6371.6967773, -16931.1562500, 16931.1542969

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.7175308
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7165810, upper bound: 9159.7165811
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8260.9833984, 6124.1367188, -8904.5888672, 6599.2202148, -14857.0205078, 15021.4072266
1: -6558.9648438, 5888.8247070, -7067.9404297, 6347.4223633, -12906.3867188, 12955.7724609
2: -9511.6953125, 6422.5688477, -10244.5087891, 6914.9130859, -16422.5664062, 16656.1679688
3: -3614.8420410, 9141.4384766, -3871.0993652, 9858.5449219, -13453.0683594, 13012.5380859
4: -10564.5058594, 6371.6967773, -11372.1064453, 6851.4018555, -17408.0605469, 17727.4003906

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6705076, upper bound: 9160.5791370
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7165810, upper bound: 9160.5780173
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8891.1933594, 6588.7363281, -8072.6899414, 5978.7265625, -14862.3964844, 14656.9589844
1: -7057.3173828, 6337.3496094, -6409.0214844, 5749.4497070, -12805.7041016, 12746.3710938
2: -10229.0732422, 6903.7670898, -9293.4628906, 6269.8530273, -16486.4003906, 16191.8808594
3: -3864.7243652, 9843.5185547, -3527.2121582, 8930.7656250, -12795.4902344, 13353.2597656
4: -11354.9492188, 6840.1601562, -10321.6591797, 6218.6733398, -17556.0117188, 17152.4453125

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5605290, upper bound: 9158.5564466
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5605290, upper bound: 9158.5618197
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8706.0839844, 6460.1113281, -9770.8320312, 7219.5048828, -15891.6298828, 16161.3339844
1: -6910.8227539, 6213.8139648, -7746.6660156, 6950.0024414, -13839.1884766, 13922.7919922
2: -10016.2490234, 6769.9584961, -11247.1806641, 7571.9580078, -17542.9296875, 17936.8496094
3: -3792.6062012, 9644.7177734, -4235.8911133, 10803.4541016, -14556.9560547, 13841.8750000
4: -11118.3271484, 6708.2817383, -12495.0800781, 7496.1127930, -18563.7773438, 19104.2578125

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5605774, upper bound: 9158.5564473
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.5605774, upper bound: 9158.9512247
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9162.4267057, upper bound: 9160.3728688
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9162.9323053, upper bound: 9160.3925932
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7188198
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7204189
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.6830114
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.6830114
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131499
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131499
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9161.1429117, upper bound: 9159.7274302
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9162.8201152, upper bound: 9159.7283536
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9161.1429117, upper bound: 9160.5899267
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9162.8201152, upper bound: 9160.5912281
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.6813391, upper bound: 9160.5848162
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.6761740
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.6761740
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063120
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063120
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.6813391
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.6813391
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.7114963
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9160.5848161, upper bound: 9160.7114963
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.7175308
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.7165810, upper bound: 9159.7165811
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.6705076, upper bound: 9160.5791370
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9159.7165810, upper bound: 9160.5780173
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -9158.5605290, upper bound: 9158.5564466
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -9158.5605290, upper bound: 9158.5618197
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -9158.5605774, upper bound: 9158.5564473
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -9158.5605774, upper bound: 9158.9512247

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5593.2919922, 4170.9360352, -5338.7373047, 3978.0178223, -9571.3095703, 9509.6728516
1: -4447.0068359, 4008.5729980, -4246.7900391, 3823.2585449, -8270.2656250, 8255.3603516
2: -6461.6699219, 4362.9394531, -6172.8627930, 4164.4814453, -10626.1513672, 10535.8017578
3: -2442.8278809, 6208.1767578, -2325.1479492, 5925.8867188, -8368.7148438, 8533.3242188
4: -7180.9687500, 4327.6752930, -6856.4863281, 4125.1684570, -11306.1367188, 11184.1611328

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4267057, upper bound: 9160.3728688
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4267057, upper bound: 9160.3728688
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5672.3452148, 4211.8769531, -5382.5039062, 4004.4062500, -9676.7509766, 9594.3789062
1: -4509.5522461, 4048.0981445, -4281.2812500, 3849.0554199, -8358.6074219, 8329.3789062
2: -6553.4570312, 4405.2534180, -6222.8759766, 4191.4501953, -10744.9042969, 10628.1279297
3: -2464.6838379, 6285.9453125, -2340.0410156, 5971.8828125, -8436.5654297, 8625.9863281
4: -7279.6313477, 4368.9628906, -6911.6005859, 4152.6235352, -11432.2548828, 11280.5615234

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
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
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9323053, upper bound: 9160.3925932
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9323053, upper bound: 9160.3925932
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6159.1230469, 4579.4897461, -9918.9531250, 10132.4433594
1: -4247.0512695, 3818.8703613, -4895.5634766, 4402.8896484, -8649.9414062, 8714.4326172
2: -6172.7690430, 4157.7558594, -7109.7338867, 4786.8471680, -10959.6162109, 11267.4902344
3: -2321.1672363, 5922.5683594, -2677.7397461, 6834.8393555, -9156.0058594, 8600.3076172
4: -6855.8618164, 4119.5634766, -7899.9721680, 4748.5356445, -11604.3964844, 12019.5351562

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7188197
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7188198
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6159.1230469, 4579.4897461, -10636.4814453, 10659.6425781
1: -4814.2382812, 4326.6323242, -4895.5634766, 4402.8896484, -9217.1259766, 9222.1953125
2: -6991.6362305, 4702.5898438, -7109.7338867, 4786.8471680, -11778.4833984, 11812.3222656
3: -2630.3791504, 6718.2563477, -2677.7397461, 6834.8393555, -9465.2177734, 9395.9960938
4: -7768.4980469, 4665.7036133, -7899.9721680, 4748.5356445, -12517.0322266, 12565.6748047

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7204189
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7204189
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -5347.4804688, 3977.3732910, -9825.8808594, 9705.1757812
1: -4652.5683594, 4191.5620117, -4253.3969727, 3823.1645508, -8475.7324219, 8444.9560547
2: -6754.8217773, 4550.6113281, -6182.0483398, 4161.4941406, -10916.3164062, 10732.6591797
3: -2527.9526367, 6502.3999023, -2321.5612793, 5932.7680664, -8460.7207031, 8823.9609375
4: -7498.1806641, 4504.6772461, -6866.1259766, 4122.8217773, -11621.0019531, 11370.8027344

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3821168, upper bound: 9157.6557325
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3837837, upper bound: 9160.3835569
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -5347.4804688, 3977.3732910, -10512.2451172, 10216.3818359
1: -5194.1127930, 4684.6015625, -4253.3969727, 3823.1645508, -9017.2763672, 8937.9980469
2: -7539.5957031, 5082.8535156, -6182.0483398, 4161.4941406, -11701.0898438, 11264.9003906
3: -2830.4941406, 7265.7309570, -2321.5612793, 5932.7680664, -8763.2617188, 9587.2890625
4: -8374.8789062, 5033.6191406, -6866.1259766, 4122.8217773, -12497.7011719, 11899.7451172

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3821168, upper bound: 9157.6557325
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3837837, upper bound: 9160.3835569
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6053.0952148, 4500.6142578, -10349.1210938, 10410.7910156
1: -4652.5683594, 4191.5620117, -4811.1455078, 4327.5795898, -8980.1464844, 9002.7060547
2: -6754.8217773, 4550.6113281, -6987.8090820, 4704.2597656, -11459.0800781, 11538.4199219
3: -2527.9526367, 6502.3999023, -2630.7399902, 6716.7734375, -9244.7265625, 9133.1396484
4: -7498.1806641, 4504.6772461, -7764.5449219, 4666.1601562, -12164.3408203, 12269.2226562

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131498
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131499
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6053.0952148, 4500.6142578, -11035.4853516, 10921.9960938
1: -5194.1127930, 4684.6015625, -4811.1455078, 4327.5795898, -9521.6904297, 9495.7460938
2: -7539.5957031, 5082.8535156, -6987.8090820, 4704.2597656, -12243.8544922, 12070.6621094
3: -2830.4941406, 7265.7309570, -2630.7399902, 6716.7734375, -9547.2675781, 9896.4677734
4: -8374.8789062, 5033.6191406, -7764.5449219, 4666.1601562, -13041.0390625, 12798.1640625

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7124299
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7124299
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -7932.6079102, 5888.1230469, -11227.5869141, 11905.9287109
1: -4247.0512695, 3818.8703613, -6298.9687500, 5661.4057617, -9908.4570312, 10117.8369141
2: -6172.7690430, 4157.7558594, -9136.0410156, 6177.9907227, -12350.7597656, 13293.7968750
3: -2321.1672363, 5922.5683594, -3478.8059082, 8781.1425781, -11102.3095703, 9401.3740234
4: -6855.8618164, 4119.5634766, -10146.0556641, 6130.1625977, -12986.0244141, 14265.6171875

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9482336, upper bound: 9159.6287591
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1429117, upper bound: 9159.6826423
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1429117, upper bound: 9159.7274302
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8239.5341797, 6108.5932617, -12165.5849609, 12740.0546875
1: -4814.2382812, 4326.6323242, -6541.9804688, 5873.8818359, -10688.1191406, 10868.6132812
2: -6991.6362305, 4702.5898438, -9486.9003906, 6406.3295898, -13397.9638672, 14189.4892578
3: -2630.3791504, 6718.2563477, -3605.9916992, 9117.7519531, -11748.1308594, 10324.2480469
4: -7768.4980469, 4665.7036133, -10536.8857422, 6355.6386719, -14124.1367188, 15202.5869141

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8344438, upper bound: 9159.6111383
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5072110, upper bound: 9159.7249636
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8201153, upper bound: 9159.6840869
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8201153, upper bound: 9159.7283536
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8594.0253906, 6377.8730469, -11717.3349609, 12567.3476562
1: -4247.0512695, 3818.8703613, -6821.5947266, 6134.5629883, -10381.6142578, 10640.4628906
2: -6172.7690430, 4157.7558594, -9889.2285156, 6685.9243164, -12858.6923828, 14046.9833984
3: -2321.1672363, 5922.5683594, -3743.8701172, 9519.6357422, -11840.8027344, 9666.4384766
4: -6855.8618164, 4119.5634766, -10976.7119141, 6625.5957031, -13481.4570312, 15096.2753906

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9534498, upper bound: 9160.4154697
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1481178, upper bound: 9160.5899265
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1481178, upper bound: 9160.5899265
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8876.2968750, 6578.6982422, -12635.6894531, 13376.8173828
1: -4814.2382812, 4326.6323242, -7045.5312500, 6327.6855469, -11141.9238281, 11372.1640625
2: -6991.6362305, 4702.5898438, -10211.9345703, 6893.5961914, -13885.2324219, 14914.5244141
3: -2630.3791504, 6718.2563477, -3859.4755859, 9827.4121094, -12457.7910156, 10577.7314453
4: -7768.4980469, 4665.7036133, -11335.8085938, 6830.2939453, -14598.7919922, 16001.5097656

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8894344, upper bound: 9160.4156034
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8313251, upper bound: 9160.5912279
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8313251, upper bound: 9160.5912279
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -7388.3930664, 5490.2221680, -11338.7314453, 11746.0878906
1: -4652.5683594, 4191.5620117, -5867.1743164, 5280.5380859, -9933.1064453, 10058.7363281
2: -6754.8217773, 4550.6113281, -8513.6015625, 5764.6777344, -12519.5000000, 13064.2128906
3: -2527.9526367, 6502.3999023, -3238.8037109, 8184.8447266, -10712.7968750, 9741.2031250
4: -7498.1806641, 4504.6772461, -9452.1464844, 5714.6831055, -13212.8632812, 13956.8242188

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3803356, upper bound: 9157.2305845
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3831987, upper bound: 9160.3822155
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -7388.3930664, 5490.2221680, -12025.0947266, 12257.2949219
1: -5194.1127930, 4684.6015625, -5867.1743164, 5280.5380859, -10474.6503906, 10551.7753906
2: -7539.5957031, 5082.8535156, -8513.6015625, 5764.6777344, -13304.2734375, 13596.4550781
3: -2830.4941406, 7265.7309570, -3238.8037109, 8184.8447266, -11015.3388672, 10504.5332031
4: -8374.8789062, 5033.6191406, -9452.1464844, 5714.6831055, -14089.5625000, 14485.7656250

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3803356, upper bound: 9157.2305845
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3831987, upper bound: 9160.3822155
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8130.4882812, 6027.3403320, -11875.8476562, 12484.5976562
1: -4652.5683594, 4191.5620117, -6455.2001953, 5795.9819336, -10448.5498047, 10646.7617188
2: -6754.8217773, 4550.6113281, -9359.7275391, 6322.8823242, -13077.7041016, 13910.3388672
3: -2527.9526367, 6502.3999023, -3559.7922363, 8995.6894531, -11523.6425781, 10062.1923828
4: -7498.1806641, 4504.6772461, -10394.4736328, 6273.7998047, -13771.9804688, 14899.1503906

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
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
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6761740, upper bound: 9159.7233580
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6761740, upper bound: 9160.5839210
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8130.4882812, 6027.3403320, -12562.2109375, 12999.3896484
1: -5194.1127930, 4684.6015625, -6455.2001953, 5795.9819336, -10990.0927734, 11139.8017578
2: -7539.5957031, 5082.8535156, -9359.7275391, 6322.8823242, -13862.4785156, 14442.5810547
3: -2830.4941406, 7265.7309570, -3559.7922363, 8995.6894531, -11826.1835938, 10825.5224609
4: -8374.8789062, 5033.6191406, -10394.4736328, 6273.7998047, -14648.6787109, 15428.0927734

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
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
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6761740, upper bound: 9159.7215790
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6761740, upper bound: 9160.5839210
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -5445.6357422, 4052.3520508, -11461.9296875, 10953.0019531
1: -5884.5581055, 5296.7612305, -4331.4916992, 3895.1152344, -9779.6738281, 9628.2529297
2: -8539.4326172, 5781.5205078, -6295.1933594, 4241.0473633, -12780.4804688, 12076.7128906
3: -3248.6958008, 8209.8476562, -2367.3967285, 6042.9243164, -9291.6201172, 10577.2441406
4: -9481.6269531, 5730.7734375, -6992.0708008, 4201.9355469, -13683.5625000, 12722.8427734

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -5445.6357422, 4052.3520508, -12222.8427734, 11504.3720703
1: -6487.3271484, 5825.9394531, -4331.4916992, 3895.1152344, -10382.4423828, 10157.4316406
2: -9407.1347656, 6354.2421875, -6295.1933594, 4241.0473633, -13648.1816406, 12649.4335938
3: -3577.6472168, 9041.6318359, -2367.3967285, 6042.9243164, -9620.5712891, 11409.0283203
4: -10448.0498047, 6304.1855469, -6992.0708008, 4201.9355469, -14649.9853516, 13296.2558594

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -6159.1230469, 4579.4897461, -11989.0683594, 11666.4882812
1: -5884.5581055, 5296.7612305, -4895.5634766, 4402.8896484, -10287.4472656, 10192.3242188
2: -8539.4326172, 5781.5205078, -7109.7338867, 4786.8471680, -13326.2792969, 12891.2529297
3: -3248.6958008, 8209.8476562, -2677.7397461, 6834.8393555, -10083.5341797, 10887.5878906
4: -9481.6269531, 5730.7734375, -7899.9721680, 4748.5356445, -14230.1621094, 13630.7441406

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063119
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063119
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -6159.1230469, 4579.4897461, -12749.9814453, 12217.8574219
1: -6487.3271484, 5825.9394531, -4895.5634766, 4402.8896484, -10890.2148438, 10721.5029297
2: -9407.1347656, 6354.2421875, -7109.7338867, 4786.8471680, -14193.9824219, 13463.9746094
3: -3577.6472168, 9041.6318359, -2677.7397461, 6834.8393555, -10412.4853516, 11719.3710938
4: -10448.0498047, 6304.1855469, -7899.9721680, 4748.5356445, -15196.5859375, 14204.1582031

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7057753
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7057753
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8139.2573242, 6046.2695312, -5347.4804688, 3977.3732910, -12116.6308594, 11393.7480469
1: -6461.4291992, 5817.6831055, -4253.3969727, 3823.1645508, -10284.5937500, 10071.0781250
2: -9369.9003906, 6341.2602539, -6182.0483398, 4161.4941406, -13531.3945312, 12523.3066406
3: -3544.0375977, 9022.6132812, -2321.5612793, 5932.7680664, -9476.8027344, 11344.1748047
4: -10397.9277344, 6278.9721680, -6866.1259766, 4122.8217773, -14520.7500000, 13145.0966797

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9718192, upper bound: 9157.6500780
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3151615, upper bound: 9160.3826478
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8814.8632812, 6533.5449219, -5347.4804688, 3977.3732910, -12784.2998047, 11881.0253906
1: -6996.9555664, 6284.2514648, -4253.3969727, 3823.1645508, -10818.1279297, 10537.6484375
2: -10141.1152344, 6846.4956055, -6182.0483398, 4161.4941406, -14302.6093750, 13028.5419922
3: -3833.6420898, 9759.3496094, -2321.5612793, 5932.7680664, -9766.4091797, 12080.9101562
4: -11256.9599609, 6783.5244141, -6866.1259766, 4122.8217773, -15379.7812500, 13649.6484375

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9718192, upper bound: 9157.6500780
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3151615, upper bound: 9160.3826478
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8139.2573242, 6046.2695312, -6053.0952148, 4500.6142578, -12639.8710938, 12099.3652344
1: -6461.4291992, 5817.6831055, -4811.1455078, 4327.5795898, -10789.0087891, 10628.8271484
2: -9369.9003906, 6341.2602539, -6987.8090820, 4704.2597656, -14074.1582031, 13329.0673828
3: -3544.0375977, 9022.6132812, -2630.7399902, 6716.7734375, -10260.8105469, 11653.3525391
4: -10397.9277344, 6278.9721680, -7764.5449219, 4666.1601562, -15064.0878906, 14043.5166016

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3376530, upper bound: 9157.5639580
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848160, upper bound: 9160.7114962
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848162, upper bound: 9160.7114962
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8814.8632812, 6533.5449219, -6053.0952148, 4500.6142578, -13315.4765625, 12586.6406250
1: -6996.9555664, 6284.2514648, -4811.1455078, 4327.5795898, -11324.5351562, 11095.3964844
2: -10141.1152344, 6846.4956055, -6987.8090820, 4704.2597656, -14845.3740234, 13834.3027344
3: -3833.6420898, 9759.3496094, -2630.7399902, 6716.7734375, -10550.4160156, 12390.0888672
4: -11256.9599609, 6783.5244141, -7764.5449219, 4666.1601562, -15923.1201172, 14548.0673828

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3376530, upper bound: 9157.5639580
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848160, upper bound: 9160.7106157
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5848162, upper bound: 9160.7106157
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -7932.6079102, 5888.1230469, -13297.7011719, 13435.2822266
1: -5884.5581055, 5296.7612305, -6298.9687500, 5661.4057617, -11545.9638672, 11595.7294922
2: -8539.4326172, 5781.5205078, -9136.0410156, 6177.9907227, -14717.4238281, 14911.3457031
3: -3248.6958008, 8209.8476562, -3478.8059082, 8781.1425781, -12021.9296875, 11684.5722656
4: -9481.6269531, 5730.7734375, -10146.0556641, 6130.1625977, -15609.0410156, 15864.4970703

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.6705076
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.7165811
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -8239.5341797, 6108.5932617, -14279.0849609, 14298.2705078
1: -6487.3271484, 5825.9394531, -6541.9804688, 5873.8818359, -12361.2080078, 12367.9199219
2: -9407.1347656, 6354.2421875, -9486.9003906, 6406.3295898, -15813.4648438, 15840.3154297
3: -3577.6472168, 9041.6318359, -3605.9916992, 9117.7519531, -12691.1923828, 12643.9082031
4: -10448.0498047, 6304.1855469, -10536.8857422, 6355.6386719, -16799.8144531, 16835.8203125

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7165811, upper bound: 9159.6705076
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7165811, upper bound: 9159.7165811
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7409.5781250, 5507.3666992, -8594.0253906, 6377.8730469, -13784.0537109, 14089.0722656
1: -5884.5581055, 5296.7612305, -6821.5947266, 6134.5629883, -12019.1201172, 12113.2851562
2: -8539.4326172, 5781.5205078, -9889.2285156, 6685.9243164, -15222.8261719, 15654.6298828
3: -3248.6958008, 8209.8476562, -3743.8701172, 9519.6357422, -12744.0537109, 11953.7158203
4: -9481.6269531, 5730.7734375, -10976.7119141, 6625.5957031, -16100.8476562, 16684.1386719

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6756395, upper bound: 9160.5780172
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6756395, upper bound: 9160.5780172
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8170.4921875, 6058.7373047, -8876.2968750, 6578.6982422, -14747.0000000, 14927.6523438
1: -6487.3271484, 5825.9394531, -7045.5312500, 6327.6855469, -12815.0126953, 12870.6806641
2: -9407.1347656, 6354.2421875, -10211.9345703, 6893.5961914, -16297.7275391, 16555.2734375
3: -3577.6472168, 9041.6318359, -3859.4755859, 9827.4121094, -13384.4677734, 12901.1064453
4: -10448.0498047, 6304.1855469, -11335.8085938, 6830.2939453, -17271.7031250, 17623.5742188

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7217735, upper bound: 9160.5780172
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7217735, upper bound: 9160.5780172
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10558.6601562, 7805.8593750, -9770.8320312, 7219.5048828, -17666.3281250, 17477.3203125
1: -8370.0732422, 7519.9335938, -7746.6660156, 6950.0024414, -15250.1748047, 15203.7822266
2: -12145.7324219, 8180.4584961, -11247.1806641, 7571.9580078, -19584.3808594, 19310.6621094
3: -4551.9360352, 11683.6074219, -4235.8911133, 10803.4541016, -15292.6132812, 15831.1406250
4: -13489.3945312, 8092.2968750, -12495.0800781, 7496.1127930, -20829.6699219, 20447.2343750

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.3694204, upper bound: 9157.3738759
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.3693591, upper bound: 9158.6751226
time: 0.95 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.34 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.4267057, upper bound: 9160.3728688
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.4267057, upper bound: 9160.3728688
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.9323053, upper bound: 9160.3925932
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.9323053, upper bound: 9160.3925932
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7188197
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7188198
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7204189
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1498402, upper bound: 9160.7204189
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3821168, upper bound: 9157.6557325
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3837837, upper bound: 9160.3835569
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3821168, upper bound: 9157.6557325
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3837837, upper bound: 9160.3835569
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131498
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7131499
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7124299
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6830114, upper bound: 9160.7124299
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1429117, upper bound: 9159.6826423
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1429117, upper bound: 9159.7274302
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.8201153, upper bound: 9159.6840869
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.8201153, upper bound: 9159.7283536
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1481178, upper bound: 9160.5899265
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9161.1481178, upper bound: 9160.5899265
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.8313251, upper bound: 9160.5912279
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9162.8313251, upper bound: 9160.5912279
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3803356, upper bound: 9157.2305845
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3831987, upper bound: 9160.3822155
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3803356, upper bound: 9157.2305845
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3831987, upper bound: 9160.3822155
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6761740, upper bound: 9159.7233580
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6761740, upper bound: 9160.5839210
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6761740, upper bound: 9159.7215790
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.6761740, upper bound: 9160.5839210
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063119
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063119
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7057753
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7057753
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.9718192, upper bound: 9157.6500780
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3151615, upper bound: 9160.3826478
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.9718192, upper bound: 9157.6500780
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.3151615, upper bound: 9160.3826478
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.5848160, upper bound: 9160.7114962
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.5848162, upper bound: 9160.7114962
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.5848160, upper bound: 9160.7106157
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9160.5848162, upper bound: 9160.7106157
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.6705076
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.7165811
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.7165811, upper bound: 9159.6705076
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.7165811, upper bound: 9159.7165811
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6756395, upper bound: 9160.5780172
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.6756395, upper bound: 9160.5780172
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.7217735, upper bound: 9160.5780172
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -9159.7217735, upper bound: 9160.5780172
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -9158.3694204, upper bound: 9157.3738759
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -9158.3693591, upper bound: 9158.6751226

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5593.2919922, 4170.9360352, -5235.3720703, 3901.2487793, -9494.5400391, 9406.3085938
1: -4447.0068359, 4008.5729980, -4164.6889648, 3749.1757812, -8196.1826172, 8173.2607422
2: -6461.6699219, 4362.9394531, -6053.7631836, 4083.7180176, -10545.3857422, 10416.7011719
3: -2442.8278809, 6208.1767578, -2280.7165527, 5809.1127930, -8251.9404297, 8488.8935547
4: -7180.9687500, 4327.6752930, -6724.0522461, 4045.2363281, -11226.2050781, 11051.7236328

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0106468, upper bound: 9158.1965342
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8487593, upper bound: 9160.2116397
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5593.2919922, 4170.9360352, -5737.2094727, 4281.0976562, -9874.3886719, 9908.1455078
1: -4447.0068359, 4008.5729980, -4564.2343750, 4117.5234375, -8564.5302734, 8572.8076172
2: -6461.6699219, 4362.9394531, -6626.9653320, 4470.7246094, -10932.3935547, 10989.9003906
3: -2442.8278809, 6208.1767578, -2484.1040039, 6381.9335938, -8824.7607422, 8692.2812500
4: -7180.9687500, 4327.6752930, -7356.4833984, 4425.3798828, -11606.3486328, 11684.1582031

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0106468, upper bound: 9158.1965342
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8487593, upper bound: 9160.2116397
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5672.3452148, 4211.8769531, -5275.0454102, 3924.0175781, -9596.3632812, 9486.9218750
1: -4509.5522461, 4048.0981445, -4195.7119141, 3771.5791016, -8281.1308594, 8243.8085938
2: -6553.4570312, 4405.2534180, -6098.9746094, 4106.7099609, -10660.1669922, 10504.2275391
3: -2464.6838379, 6285.9453125, -2292.7272949, 5849.7939453, -8314.4775391, 8578.6728516
4: -7279.6313477, 4368.9628906, -6773.7114258, 4068.8820801, -11348.5136719, 11142.6728516

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3886888
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3925932
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5672.3452148, 4211.8769531, -5778.4179688, 4304.0327148, -9976.3769531, 9990.2929688
1: -4509.5522461, 4048.0981445, -4597.0639648, 4139.6972656, -8649.2500000, 8645.1611328
2: -6553.4570312, 4405.2534180, -6675.0786133, 4494.9877930, -11048.4414062, 11080.3320312
3: -2464.6838379, 6285.9453125, -2497.4204102, 6422.7666016, -8887.4492188, 8783.3652344
4: -7279.6313477, 4368.9628906, -7409.5122070, 4449.8686523, -11729.5000000, 11778.4746094

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3886888
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3925932
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6056.9916992, 4500.5209961, -9839.9853516, 10030.3134766
1: -4247.0512695, 3818.8703613, -4814.2382812, 4326.6323242, -8573.6835938, 8633.1074219
2: -6172.7690430, 4157.7558594, -6991.6362305, 4702.5898438, -10875.3574219, 11149.3925781
3: -2321.1672363, 5922.5683594, -2630.3791504, 6718.2563477, -9039.4238281, 8552.9472656
4: -6855.8618164, 4119.5634766, -7768.4980469, 4665.7036133, -11521.5634766, 11888.0605469

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.7838877, upper bound: 9160.4313914
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7229784, upper bound: 9160.4506295
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6521.0463867, 4861.1459961, -10200.6103516, 10494.3691406
1: -4247.0512695, 3818.8703613, -5183.4223633, 4677.0102539, -8924.0585938, 9002.2919922
2: -6172.7690430, 4157.7558594, -7524.6733398, 5074.9497070, -11247.7187500, 11682.4287109
3: -2321.1672363, 5922.5683594, -2826.6457520, 7252.5444336, -9573.7119141, 8749.2138672
4: -6855.8618164, 4119.5634766, -8358.0966797, 5025.9990234, -11881.8603516, 12477.6591797

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.7838877, upper bound: 9160.4313914
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7229784, upper bound: 9160.4506295
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6056.9916992, 4500.5209961, -10557.5126953, 10557.5126953
1: -4814.2382812, 4326.6323242, -4814.2382812, 4326.6323242, -9140.8691406, 9140.8701172
2: -6991.6362305, 4702.5898438, -6991.6362305, 4702.5898438, -11694.2265625, 11694.2255859
3: -2630.3791504, 6718.2563477, -2630.3791504, 6718.2563477, -9348.6357422, 9348.6357422
4: -7768.4980469, 4665.7036133, -7768.4980469, 4665.7036133, -12434.2001953, 12434.1992188

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0398818, upper bound: 9160.7204189
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8955752, upper bound: 9160.7204189
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6521.0463867, 4861.1459961, -10918.1376953, 11021.5673828
1: -4814.2382812, 4326.6323242, -5183.4223633, 4677.0102539, -9491.2451172, 9510.0546875
2: -6991.6362305, 4702.5898438, -7524.6733398, 5074.9497070, -12066.5859375, 12227.2617188
3: -2630.3791504, 6718.2563477, -2826.6457520, 7252.5444336, -9882.9238281, 9544.9023438
4: -7768.4980469, 4665.7036133, -8358.0966797, 5025.9990234, -12794.4960938, 13023.7988281

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0398818, upper bound: 9160.7204189
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8955752, upper bound: 9160.7204189
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5737.2094727, 4281.0976562, -5007.9814453, 3737.1994629, -9474.4082031, 9289.0781250
1: -4564.2343750, 4117.5234375, -3984.0983887, 3590.8010254, -8155.0351562, 8101.6220703
2: -6626.9653320, 4470.7246094, -5793.0961914, 3910.1442871, -10537.1083984, 10263.8203125
3: -2484.1040039, 6381.9335938, -2181.9057617, 5559.2910156, -8043.3945312, 8563.8388672
4: -7356.4833984, 4425.3798828, -6436.1884766, 3872.8337402, -11229.3173828, 10861.5683594

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1493748, upper bound: 9156.5363204
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.2330711, upper bound: 9156.5362967
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5778.4179688, 4304.0327148, -5199.8745117, 3865.1872559, -9643.6054688, 9503.9062500
1: -4597.0639648, 4139.6972656, -4136.2744141, 3714.9035645, -8311.9667969, 8275.9716797
2: -6675.0786133, 4494.9877930, -6013.0346680, 4044.6931152, -10719.7714844, 10508.0205078
3: -2497.4204102, 6422.7666016, -2257.4101562, 5766.4951172, -8263.9160156, 8680.1748047
4: -7409.5122070, 4449.8686523, -6678.1660156, 4007.2468262, -11416.7587891, 11128.0351562

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1511412, upper bound: 9158.2347178
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.2347166, upper bound: 9158.2347159
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6443.5483398, 4803.4741211, -5007.9814453, 3737.1994629, -10180.7441406, 9811.4550781
1: -5121.8295898, 4621.1962891, -3984.0983887, 3590.8010254, -8712.6279297, 8605.2949219
2: -7434.7172852, 5014.2924805, -5793.0961914, 3910.1442871, -11344.8613281, 10807.3886719
3: -2792.9338379, 7163.2783203, -2181.9057617, 5559.2910156, -8352.2246094, 9345.1826172
4: -8258.5244141, 4965.8959961, -6436.1884766, 3872.8337402, -12131.3574219, 11402.0839844

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2139037, upper bound: 9156.5354358
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13

Time for candidate selection: 9.49 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3913687, upper bound: 9152.7609594
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.6523536, upper bound: 9152.7600012
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6447.3623047, 4805.0415039, -5199.8745117, 3865.1872559, -10312.5498047, 10004.9140625
1: -5124.7148438, 4623.2246094, -4136.2744141, 3714.9035645, -8839.6181641, 8759.4990234
2: -7440.0000000, 5017.1093750, -6013.0346680, 4044.6931152, -11484.6933594, 11030.1435547
3: -2794.4968262, 7170.5463867, -2257.4101562, 5766.4951172, -8560.9902344, 9427.9541016
4: -8264.3369141, 4968.6020508, -6678.1660156, 4007.2468262, -12271.5810547, 11646.7675781

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2709105, upper bound: 9158.2341794
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1569344, upper bound: 9160.3717633
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4049494, upper bound: 9160.3745747
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6056.9916992, 4500.5209961, -10349.0302734, 10414.6875000
1: -4652.5683594, 4191.5620117, -4814.2382812, 4326.6323242, -8979.2001953, 9005.7978516
2: -6754.8217773, 4550.6113281, -6991.6362305, 4702.5898438, -11457.4111328, 11542.2480469
3: -2527.9526367, 6502.3999023, -2630.3791504, 6718.2563477, -9246.2089844, 9132.7792969
4: -7498.1806641, 4504.6772461, -7768.4980469, 4665.7036133, -12163.8818359, 12273.1757812

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6557325, upper bound: 9160.4193115
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3835569, upper bound: 9160.4472813
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -6521.0463867, 4861.1459961, -10709.6562500, 10878.7421875
1: -4652.5683594, 4191.5620117, -5183.4223633, 4677.0102539, -9329.5751953, 9374.9824219
2: -6754.8217773, 4550.6113281, -7524.6733398, 5074.9497070, -11829.7714844, 12075.2851562
3: -2527.9526367, 6502.3999023, -2826.6457520, 7252.5444336, -9780.4970703, 9329.0458984
4: -7498.1806641, 4504.6772461, -8358.0966797, 5025.9990234, -12524.1787109, 12862.7734375

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6557325, upper bound: 9160.4193115
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3835569, upper bound: 9160.4472813
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6056.9916992, 4500.5209961, -11035.3935547, 10925.8925781
1: -5194.1127930, 4684.6015625, -4814.2382812, 4326.6323242, -9520.7441406, 9498.8388672
2: -7539.5957031, 5082.8535156, -6991.6362305, 4702.5898438, -12242.1855469, 12074.4902344
3: -2830.4941406, 7265.7309570, -2630.3791504, 6718.2563477, -9548.7500000, 9896.1093750
4: -8374.8789062, 5033.6191406, -7768.4980469, 4665.7036133, -13040.5820312, 12802.1171875

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1969647, upper bound: 9160.6545018
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6587743, upper bound: 9160.6587744
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -6521.0463867, 4861.1459961, -11396.0195312, 11389.9472656
1: -5194.1127930, 4684.6015625, -5183.4223633, 4677.0102539, -9871.1191406, 9868.0224609
2: -7539.5957031, 5082.8535156, -7524.6733398, 5074.9497070, -12614.5449219, 12607.5263672
3: -2830.4941406, 7265.7309570, -2826.6457520, 7252.5444336, -10083.0390625, 10092.3759766
4: -8374.8789062, 5033.6191406, -8358.0966797, 5025.9990234, -13400.8779297, 13391.7158203

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1969649, upper bound: 9160.6545018
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6587743, upper bound: 9160.6587744
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -7409.5781250, 5507.3666992, -10846.8310547, 11382.8994141
1: -4247.0512695, 3818.8703613, -5884.5581055, 5296.7612305, -9543.8115234, 9703.4267578
2: -6172.7690430, 4157.7558594, -8539.4326172, 5781.5205078, -11954.2880859, 12697.1884766
3: -2321.1672363, 5922.5683594, -3248.6958008, 8209.8476562, -10531.0146484, 9171.2636719
4: -6855.8618164, 4119.5634766, -9481.6269531, 5730.7734375, -12586.6328125, 13601.1904297

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
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

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.7600964, upper bound: 9159.4117493
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7166620, upper bound: 9159.4197208
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8170.4921875, 6058.7373047, -11398.2001953, 12143.8134766
1: -4247.0512695, 3818.8703613, -6487.3271484, 5825.9394531, -10072.9902344, 10306.1943359
2: -6172.7690430, 4157.7558594, -9407.1347656, 6354.2421875, -12527.0097656, 13564.8906250
3: -2321.1672363, 5922.5683594, -3577.6472168, 9041.6318359, -11362.7988281, 9500.2158203
4: -6855.8618164, 4119.5634766, -10448.0498047, 6304.1855469, -13160.0468750, 14567.6123047

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.7600964, upper bound: 9159.4304912
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7166620, upper bound: 9159.6043557
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -7409.5781250, 5507.3666992, -11564.3583984, 11910.0996094
1: -4814.2382812, 4326.6323242, -5884.5581055, 5296.7612305, -10110.9980469, 10211.1904297
2: -6991.6362305, 4702.5898438, -8539.4326172, 5781.5205078, -12773.1562500, 13242.0224609
3: -2630.3791504, 6718.2563477, -3248.6958008, 8209.8476562, -10840.2265625, 9966.9521484
4: -7768.4980469, 4665.7036133, -9481.6269531, 5730.7734375, -13499.2695312, 14147.3300781

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8016024, upper bound: 9159.6840869
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8201152, upper bound: 9159.6840869
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8170.4921875, 6058.7373047, -12115.7285156, 12671.0126953
1: -4814.2382812, 4326.6323242, -6487.3271484, 5825.9394531, -10640.1777344, 10813.9580078
2: -6991.6362305, 4702.5898438, -9407.1347656, 6354.2421875, -13345.8769531, 14109.7246094
3: -2630.3791504, 6718.2563477, -3577.6472168, 9041.6318359, -11672.0107422, 10295.9033203
4: -7768.4980469, 4665.7036133, -10448.0498047, 6304.1855469, -14072.6835938, 15113.7519531

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8016024, upper bound: 9159.7282700
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8201153, upper bound: 9159.7283536
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8139.2573242, 6046.2695312, -11385.7324219, 12112.5791016
1: -4247.0512695, 3818.8703613, -6461.4291992, 5817.6831055, -10064.7324219, 10280.2978516
2: -6172.7690430, 4157.7558594, -9369.9003906, 6341.2602539, -12514.0263672, 13527.6562500
3: -2321.1672363, 5922.5683594, -3544.0375977, 9022.6132812, -11343.7802734, 9466.6044922
4: -6855.8618164, 4119.5634766, -10397.9277344, 6278.9721680, -13134.8339844, 14517.4912109

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.8126745, upper bound: 9160.0679657
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7220784, upper bound: 9160.3187442
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8814.8632812, 6533.5449219, -11873.0078125, 12780.6728516
1: -4247.0512695, 3818.8703613, -6996.9555664, 6284.2514648, -10531.3017578, 10814.2724609
2: -6172.7690430, 4157.7558594, -10141.1152344, 6846.4956055, -13019.2626953, 14298.8710938
3: -2321.1672363, 5922.5683594, -3833.6420898, 9759.3496094, -12080.5166016, 9756.2109375
4: -6855.8618164, 4119.5634766, -11256.9599609, 6783.5244141, -13639.3837891, 15376.5224609

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.8126745, upper bound: 9160.0679657
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7220784, upper bound: 9160.3319690
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8139.2573242, 6046.2695312, -12103.2597656, 12639.7783203
1: -4814.2382812, 4326.6323242, -6461.4291992, 5817.6831055, -10631.9189453, 10788.0615234
2: -6991.6362305, 4702.5898438, -9369.9003906, 6341.2602539, -13332.8964844, 14072.4902344
3: -2630.3791504, 6718.2563477, -3544.0375977, 9022.6132812, -11652.9921875, 10262.2929688
4: -7768.4980469, 4665.7036133, -10397.9277344, 6278.9721680, -14047.4697266, 15063.6289062

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5357657, upper bound: 9157.3431340
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8165871, upper bound: 9160.5912279
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8313250, upper bound: 9160.5912279
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8814.8632812, 6533.5449219, -12590.5361328, 13315.3847656
1: -4814.2382812, 4326.6323242, -6996.9555664, 6284.2514648, -11098.4882812, 11323.5878906
2: -6991.6362305, 4702.5898438, -10141.1152344, 6846.4956055, -13838.1318359, 14843.7050781
3: -2630.3791504, 6718.2563477, -3833.6420898, 9759.3496094, -12389.7285156, 10551.8984375
4: -7768.4980469, 4665.7036133, -11256.9599609, 6783.5244141, -14552.0195312, 15922.6611328

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5357660, upper bound: 9157.3431340
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8165871, upper bound: 9160.5912279
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8313251, upper bound: 9160.5912279
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5737.2094727, 4281.0976562, -7165.0961914, 5331.1538086, -11068.3623047, 11446.1923828
1: -4564.2343750, 4117.5234375, -5689.8852539, 5126.3222656, -9690.5566406, 9807.4082031
2: -6626.9653320, 4470.7246094, -8257.3837891, 5598.3007812, -12225.2646484, 12728.1083984
3: -2484.1040039, 6381.9335938, -3147.4995117, 7936.6577148, -10420.7617188, 9529.4306641
4: -7356.4833984, 4425.3798828, -9168.3388672, 5549.4829102, -12905.9648438, 13593.7187500

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.4089131, upper bound: 9155.3829528
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1431117, upper bound: 9156.1267772
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.2312494, upper bound: 9156.1267553
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5778.4179688, 4304.0327148, -7075.8618164, 5266.3232422, -11044.7412109, 11379.8945312
1: -4597.0639648, 4139.6972656, -5619.2846680, 5064.7285156, -9661.7929688, 9758.9824219
2: -6675.0786133, 4494.9877930, -8155.8544922, 5533.3261719, -12208.4042969, 12650.8417969
3: -2497.4204102, 6422.7666016, -3111.3366699, 7842.5756836, -10339.9960938, 9534.1035156
4: -7409.5122070, 4449.8686523, -9055.3232422, 5485.2158203, -12894.7285156, 13505.1914062

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.4115301, upper bound: 9157.8678219
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1496936, upper bound: 9157.9499837
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.2333226, upper bound: 9157.9499771
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6443.5483398, 4803.4741211, -7165.0961914, 5331.1538086, -11774.6992188, 11966.9082031
1: -5121.8295898, 4621.1962891, -5689.8852539, 5126.3222656, -10248.1494141, 10311.0820312
2: -7434.7172852, 5014.2924805, -8257.3837891, 5598.3007812, -13033.0175781, 13271.6757812
3: -2792.9338379, 7163.2783203, -3147.4995117, 7936.6577148, -10729.5917969, 10310.7753906
4: -8258.5244141, 4965.8959961, -9168.3388672, 5549.4829102, -13808.0048828, 14134.2343750

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.4142798, upper bound: 9155.3220864
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1828158, upper bound: 9156.1259677
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 13

Time for candidate selection: 10.85 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1067491, upper bound: 9151.1125384
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.5378699, upper bound: 9151.1113803
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6447.3623047, 4805.0415039, -7075.8618164, 5266.3232422, -11713.6855469, 11880.9033203
1: -5124.7148438, 4623.2246094, -5619.2846680, 5064.7285156, -10189.4433594, 10242.5097656
2: -7440.0000000, 5017.1093750, -8155.8544922, 5533.3261719, -12973.3261719, 13172.9638672
3: -2794.4968262, 7170.5463867, -3111.3366699, 7842.5756836, -10637.0712891, 10281.8828125
4: -8264.3369141, 4968.6020508, -9055.3232422, 5485.2158203, -13749.5527344, 14023.9248047

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.4806884, upper bound: 9157.8653822
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2694979, upper bound: 9157.9492772
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1566028, upper bound: 9160.3130794
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4043647, upper bound: 9160.3159967
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8170.4921875, 6058.7373047, -11907.2451172, 12523.7314453
1: -4652.5683594, 4191.5620117, -6487.3271484, 5825.9394531, -10478.5078125, 10678.8867188
2: -6754.8217773, 4550.6113281, -9407.1347656, 6354.2421875, -13109.0615234, 13957.7460938
3: -2527.9526367, 6502.3999023, -3577.6472168, 9041.6318359, -11569.5839844, 10080.0468750
4: -7498.1806641, 4504.6772461, -10448.0498047, 6304.1855469, -13802.3662109, 14952.7265625

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9157.6484694, upper bound: 9159.4251516
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3772680, upper bound: 9159.6018512
time: 0.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.71 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.0106468, upper bound: 9158.1965342
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9161.8487593, upper bound: 9160.2116397
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.0106468, upper bound: 9158.1965342
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9161.8487593, upper bound: 9160.2116397
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3886888
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3925932
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3886888
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7233692, upper bound: 9160.3925932
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9158.7838877, upper bound: 9160.4313914
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7229784, upper bound: 9160.4506295
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9158.7838877, upper bound: 9160.4313914
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7229784, upper bound: 9160.4506295
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9163.0398818, upper bound: 9160.7204189
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8955752, upper bound: 9160.7204189
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9163.0398818, upper bound: 9160.7204189
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8955752, upper bound: 9160.7204189
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1493748, upper bound: 9156.5363204
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.71
Output dim: 0, lower bound: -9158.2330711, upper bound: 9156.5362967
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1511412, upper bound: 9158.2347178
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.71
Output dim: 0, lower bound: -9158.2347166, upper bound: 9158.2347159
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.3913687, upper bound: 9152.7609594
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.71
Output dim: 0, lower bound: -9156.6523536, upper bound: 9152.7600012
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1569344, upper bound: 9160.3717633
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.4049494, upper bound: 9160.3745747
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.6557325, upper bound: 9160.4193115
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.3835569, upper bound: 9160.4472813
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.6557325, upper bound: 9160.4193115
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.3835569, upper bound: 9160.4472813
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1969647, upper bound: 9160.6545018
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.6587743, upper bound: 9160.6587744
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1969649, upper bound: 9160.6545018
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.6587743, upper bound: 9160.6587744
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.7600964, upper bound: 9159.4117493
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7166620, upper bound: 9159.4197208
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.7600964, upper bound: 9159.4304912
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7166620, upper bound: 9159.6043557
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8016024, upper bound: 9159.6840869
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8201152, upper bound: 9159.6840869
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8016024, upper bound: 9159.7282700
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8201153, upper bound: 9159.7283536
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.8126745, upper bound: 9160.0679657
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7220784, upper bound: 9160.3187442
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.8126745, upper bound: 9160.0679657
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.7220784, upper bound: 9160.3319690
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8165871, upper bound: 9160.5912279
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8313250, upper bound: 9160.5912279
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8165871, upper bound: 9160.5912279
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9162.8313251, upper bound: 9160.5912279
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1431117, upper bound: 9156.1267772
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.71
Output dim: 0, lower bound: -9158.2312494, upper bound: 9156.1267553
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1496936, upper bound: 9157.9499837
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.71
Output dim: 0, lower bound: -9158.2333226, upper bound: 9157.9499771
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1067491, upper bound: 9151.1125384
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.71
Output dim: 0, lower bound: -9156.5378699, upper bound: 9151.1113803
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.1566028, upper bound: 9160.3130794
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.4043647, upper bound: 9160.3159967
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9157.6484694, upper bound: 9159.4251516
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.71
Output dim: 0, lower bound: -9160.3772680, upper bound: 9159.6018512
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.6761740, upper bound: 9160.5839210
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.6761740, upper bound: 9159.7215790
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.6761740, upper bound: 9160.5839210
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6790497, upper bound: 9160.6913185
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063119
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7063119
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7057753
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6773130, upper bound: 9160.7057753
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.9718192, upper bound: 9157.6500780
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.3151615, upper bound: 9160.3826478
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.9718192, upper bound: 9157.6500780
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.3151615, upper bound: 9160.3826478
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.5848160, upper bound: 9160.7114962
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.5848162, upper bound: 9160.7114962
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.5848160, upper bound: 9160.7106157
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9160.5848162, upper bound: 9160.7106157
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.6705076
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6705076, upper bound: 9159.7165811
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.7165811, upper bound: 9159.6705076
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.7165811, upper bound: 9159.7165811
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6756395, upper bound: 9160.5780172
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.6756395, upper bound: 9160.5780172
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.7217735, upper bound: 9160.5780172
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 0, lower bound: -9159.7217735, upper bound: 9160.5780172
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=11463.63671875
rel_dist={0: [-9164.409543284748, 9164.409543284746]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0579898, upper bound: 9162.6304757
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6245364, upper bound: 9162.6245364
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -9164.0579898, upper bound: 9162.6304757
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -9162.6245364, upper bound: 9162.6245364

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -6437.1933594, 4788.8964844, -11086.9345703, 11118.5976562
1: -5006.5761719, 4500.5805664, -5117.1977539, 4604.0869141, -9610.6630859, 9617.7783203
2: -7271.7465820, 4893.0332031, -7431.0668945, 5009.2968750, -12281.0429688, 12324.0996094
3: -2736.1677246, 6988.3081055, -2801.2868652, 7143.8032227, -9879.9707031, 9789.5947266
4: -8079.3994141, 4853.4165039, -8256.2177734, 4968.4692383, -13047.8691406, 13109.6347656

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6245364, upper bound: 9162.6245364
time: 0.85 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6245364, upper bound: 9162.6245364
time: 1.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8358.9042969, 6198.2509766, -6370.9326172, 4742.7211914, -13101.6240234, 12569.1835938
1: -6636.6801758, 5959.9243164, -5064.7001953, 4559.4702148, -11196.1484375, 11024.6250000
2: -9623.8066406, 6501.3457031, -7354.2685547, 4963.1645508, -14586.9697266, 13855.6123047
3: -3659.6928711, 9251.1083984, -2777.6762695, 7069.6499023, -10729.3427734, 12028.7851562
4: -10688.8964844, 6450.9218750, -8170.7900391, 4923.3544922, -15612.2509766, 14621.7099609

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1885037, upper bound: 9162.4173950
time: 0.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1876472, upper bound: 9162.1876472
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -9162.6245364, upper bound: 9162.6245364
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -9162.6245364, upper bound: 9162.6245364
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -9162.1885037, upper bound: 9162.4173950
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -9162.1876472, upper bound: 9162.1876472

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -6298.0380859, 4681.4052734, -10979.4433594, 10979.4433594
1: -5006.5761719, 4500.5805664, -5006.5761719, 4500.5805664, -9507.1562500, 9507.1562500
2: -7271.7465820, 4893.0332031, -7271.7465820, 4893.0332031, -12164.7792969, 12164.7792969
3: -2736.1677246, 6988.3081055, -2736.1677246, 6988.3081055, -9724.4755859, 9724.4755859
4: -8079.3994141, 4853.4165039, -8079.3994141, 4853.4165039, -12932.8164062, 12932.8164062

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0247485, upper bound: 9161.5694782
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7255910, upper bound: 9161.5645320
time: 0.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6298.0380859, 4681.4052734, -8358.9042969, 6198.2509766, -12496.2890625, 13040.3066406
1: -5006.5761719, 4500.5805664, -6636.6801758, 5959.9243164, -10966.5000000, 11137.2597656
2: -7271.7465820, 4893.0332031, -9623.8066406, 6501.3457031, -13773.0917969, 14516.8378906
3: -2736.1677246, 6988.3081055, -3659.6928711, 9251.1083984, -11987.2763672, 10648.0009766
4: -8079.3994141, 4853.4165039, -10688.8964844, 6450.9218750, -14530.3193359, 15542.3125000

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0247485, upper bound: 9161.5694782
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7255910, upper bound: 9161.5645320
time: 0.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7832.1831055, 5821.5581055, -5454.9301758, 4063.7148438, -11895.8984375, 11276.4882812
1: -6219.5991211, 5597.0615234, -4338.7314453, 3905.7001953, -10125.2988281, 9935.7929688
2: -9021.5615234, 6110.7192383, -6305.3774414, 4255.1699219, -13276.7314453, 12416.0966797
3: -3442.1701660, 8674.3730469, -2376.6437988, 6054.3037109, -9496.4736328, 11051.0166016
4: -10017.9218750, 6064.5170898, -7003.1943359, 4216.6855469, -14234.6074219, 13067.7109375

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8603219, upper bound: 9159.7586355
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138745, upper bound: 9159.7579009
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8307.9726562, 6161.2651367, -6220.9526367, 4633.6826172, -12941.6523438, 12382.2177734
1: -6596.3754883, 5924.3256836, -4945.1333008, 4455.0097656, -11051.3847656, 10869.4589844
2: -9564.9785156, 6462.7324219, -7179.7934570, 4849.9785156, -14414.9560547, 13642.5253906
3: -3638.7329102, 9194.8076172, -2715.8061523, 6905.1074219, -10543.8398438, 11910.6132812
4: -10623.3681641, 6412.7973633, -7977.3964844, 4811.6430664, -15435.0097656, 14390.1933594

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8590785, upper bound: 9159.7134072
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127394, upper bound: 9159.7127398
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9163.0247485, upper bound: 9161.5694782
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9161.7255910, upper bound: 9161.5645320
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9163.0247485, upper bound: 9161.5694782
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9161.7255910, upper bound: 9161.5645320
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9159.8603219, upper bound: 9159.7586355
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9159.7138745, upper bound: 9159.7579009
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9159.8590785, upper bound: 9159.7134072
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -9159.7127394, upper bound: 9159.7127398

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -6237.9121094, 4635.7104492, -10806.7656250, 10822.4003906
1: -4905.5644531, 4407.1103516, -4958.7387695, 4456.5341797, -9362.0976562, 9365.8457031
2: -7125.1625977, 4789.8227539, -7202.3852539, 4844.3603516, -11969.5234375, 11992.2070312
3: -2678.3530273, 6844.5351562, -2708.9260254, 6920.3642578, -9598.7167969, 9553.4609375
4: -7916.1811523, 4751.7998047, -8002.1567383, 4805.4936523, -12721.6748047, 12753.9560547

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7002755, upper bound: 9160.7014370
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3540765, upper bound: 9160.7148390
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -6120.9736328, 4546.7006836, -11198.9335938, 11074.3779297
1: -5287.2490234, 4766.0849609, -4865.7070312, 4371.9843750, -9659.2333984, 9631.7919922
2: -7675.0820312, 5170.7749023, -7067.5332031, 4751.6914062, -12426.7734375, 12238.3066406
3: -2878.7878418, 7395.5224609, -2655.5766602, 6789.1625977, -9667.9501953, 10051.0996094
4: -8525.3037109, 5120.5253906, -7852.0966797, 4712.8378906, -13238.1416016, 12972.6220703

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7100174, upper bound: 9160.6957119
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7099159, upper bound: 9160.7099159
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6171.0551758, 4584.4877930, -8312.5039062, 6162.9921875, -12334.0468750, 12896.9921875
1: -4905.5644531, 4407.1103516, -6599.8701172, 5926.0932617, -10831.6562500, 11006.9804688
2: -7125.1625977, 4789.8227539, -9570.6650391, 6463.9350586, -13589.0966797, 14360.4873047
3: -2678.3530273, 6844.5351562, -3638.5148926, 9199.0175781, -11877.3710938, 10483.0498047
4: -7916.1811523, 4751.7998047, -10629.9003906, 6413.3403320, -14329.5214844, 15381.6992188

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4764455, upper bound: 9160.7134094
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9869521, upper bound: 9160.5856011
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6652.2324219, 4953.4042969, -8165.9638672, 6050.0932617, -12702.3261719, 13119.3681641
1: -5287.2490234, 4766.0849609, -6482.9887695, 5818.0986328, -11105.3476562, 11249.0732422
2: -7675.0820312, 5170.7749023, -9400.5605469, 6346.5429688, -14021.6240234, 14571.3359375
3: -2878.7878418, 7395.5224609, -3570.9396973, 9033.2060547, -11911.9941406, 10966.4619141
4: -8525.3037109, 5120.5253906, -10439.6982422, 6296.6450195, -14821.9492188, 15560.2236328

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7087565, upper bound: 9160.7089435
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7083279, upper bound: 9160.5820856
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7668.4628906, 5695.8222656, -5332.3769531, 3972.7214355, -11641.1835938, 11028.1992188
1: -6089.5400391, 5476.5043945, -4241.4550781, 3818.2451172, -9907.7841797, 9717.9589844
2: -8832.9228516, 5977.6767578, -6164.1635742, 4159.0395508, -12991.9628906, 12141.8398438
3: -3366.0034180, 8492.4472656, -2323.6337891, 5918.7519531, -9284.7539062, 10816.0791016
4: -9808.8798828, 5931.0073242, -6846.3803711, 4121.9208984, -13930.8007812, 12777.3847656

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6810499, upper bound: 9159.7132641
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6810499, upper bound: 9159.7579009
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9350.4433594, 6922.4003906, -5308.8208008, 3955.2929688, -13251.1689453, 12231.2187500
1: -7414.0415039, 6663.9531250, -4223.3872070, 3801.2980957, -11186.7841797, 10887.3398438
2: -10767.2695312, 7265.3481445, -6137.6401367, 4141.3593750, -14857.4335938, 13402.9882812
3: -4067.5166016, 10346.7041016, -2314.7973633, 5893.4223633, -9960.9394531, 12650.6494141
4: -11960.8769531, 7191.9321289, -6816.3774414, 4104.8876953, -16005.2031250, 14008.3095703

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7132641
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7579009
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8160.6416016, 6048.0356445, -6110.1933594, 4548.8559570, -12709.4970703, 12158.2285156
1: -6479.2739258, 5815.7602539, -4857.0024414, 4373.5161133, -10852.7900391, 10672.7607422
2: -9395.2343750, 6342.2177734, -7051.8203125, 4759.5781250, -14154.8125000, 13394.0380859
3: -3569.1938477, 9031.1269531, -2665.6557617, 6781.2192383, -10350.4130859, 11696.7822266
4: -10435.3886719, 6291.0380859, -7835.2441406, 4722.3837891, -15157.7724609, 14126.2822266

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
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
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6798520, upper bound: 9159.6798520
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6798520, upper bound: 9159.7127398
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9807.6503906, 7251.0117188, -6040.6010742, 4505.1723633, -14263.7275391, 13291.6132812
1: -7776.5473633, 6979.5214844, -4802.5834961, 4331.6401367, -12083.0927734, 11782.1054688
2: -11290.4785156, 7604.8266602, -6972.5151367, 4715.8876953, -15962.7832031, 14577.3417969
3: -4256.6762695, 10846.8496094, -2643.0109863, 6710.1469727, -10966.8232422, 13482.1806641
4: -12543.2353516, 7529.3774414, -7746.8955078, 4679.6713867, -17170.8886719, 15276.2724609

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.6798520
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.7127398
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9162.7002755, upper bound: 9160.7014370
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9162.3540765, upper bound: 9160.7148390
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9160.7100174, upper bound: 9160.6957119
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9160.7099159, upper bound: 9160.7099159
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9162.4764455, upper bound: 9160.7134094
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9161.9869521, upper bound: 9160.5856011
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9160.7087565, upper bound: 9160.7089435
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9160.7083279, upper bound: 9160.5820856
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.6810499, upper bound: 9159.7132641
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.6810499, upper bound: 9159.7579009
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7132641
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7579009
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.6798520, upper bound: 9159.6798520
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.6798520, upper bound: 9159.7127398
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.6798520
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.11
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.7127398

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5711.9584961, 4249.7299805, -5404.1401367, 4021.5253906, -9733.4824219, 9653.8701172
1: -4541.9575195, 4084.6984863, -4298.4575195, 3865.3977051, -8407.3554688, 8383.1552734
2: -6599.2260742, 4444.7661133, -6247.3911133, 4208.5478516, -10807.7734375, 10692.1572266
3: -2484.9694824, 6336.4218750, -2349.0993652, 5995.9501953, -8480.9199219, 8685.5214844
4: -7330.4633789, 4407.6513672, -6938.8808594, 4169.7919922, -11500.2558594, 11346.5312500

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2858128, upper bound: 9159.0587247
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3993583, upper bound: 9160.3869796
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6105.0683594, 4536.0356445, -6116.9072266, 4546.7216797, -10651.7880859, 10652.9414062
1: -4852.3642578, 4360.7006836, -4861.9399414, 4371.2749023, -9223.6386719, 9222.6406250
2: -7047.3994141, 4739.4560547, -7060.9272461, 4751.9179688, -11799.3173828, 11800.3828125
3: -2650.5458984, 6771.7993164, -2658.0998535, 6786.5708008, -9437.1171875, 9429.8994141
4: -7830.4267578, 4702.0278320, -7845.5717773, 4714.1811523, -12544.6074219, 12547.5986328

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7120745
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7148390
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6166.5722656, 4599.8256836, -5284.9296875, 3929.3693848, -10095.9414062, 9884.7519531
1: -4904.3017578, 4425.1694336, -4203.6108398, 3776.9763184, -8681.2773438, 8628.7802734
2: -7120.4892578, 4803.7558594, -6109.9667969, 4110.3730469, -11230.8623047, 10913.7226562
3: -2674.7241211, 6859.6616211, -2292.3381348, 5862.0122070, -8536.7353516, 9152.0000000
4: -7905.9101562, 4758.4199219, -6785.8813477, 4072.1474609, -11978.0576172, 11544.3007812

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9582995, upper bound: 9159.2345030
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.6802222
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.6802222
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6583.9965820, 4904.4194336, -5982.1625977, 4447.2451172, -11031.2392578, 10886.5820312
1: -5233.0166016, 4718.8378906, -4754.7382812, 4276.6386719, -9509.6552734, 9473.5742188
2: -7596.3549805, 5119.8242188, -6906.1962891, 4648.2451172, -12244.5996094, 12026.0185547
3: -2850.9125977, 7320.0991211, -2598.8793945, 6637.5014648, -9488.4140625, 9918.9775391
4: -8438.0537109, 5070.2192383, -7673.7441406, 4610.4365234, -13048.4882812, 12743.9628906

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7099159
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7099159
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5711.9584961, 4249.7299805, -7469.9404297, 5552.6489258, -11264.6074219, 11719.6689453
1: -4541.9575195, 4084.6984863, -5932.4291992, 5340.2856445, -9882.2402344, 10017.1269531
2: -6599.2260742, 4444.7661133, -8608.5683594, 5829.5302734, -12428.7558594, 13053.3320312
3: -2484.9694824, 6336.4218750, -3275.7890625, 8277.1914062, -10762.1611328, 9612.2109375
4: -7330.4633789, 4407.6513672, -9558.3632812, 5778.6982422, -13109.1621094, 13966.0126953

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6737074, upper bound: 9158.3550153
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1768476, upper bound: 9160.3844795
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6105.0683594, 4536.0356445, -8221.8310547, 6097.3745117, -12202.4423828, 12757.8671875
1: -4852.3642578, 4360.7006836, -6528.1015625, 5862.9794922, -10715.3437500, 10888.8027344
2: -7047.3994141, 4739.4560547, -9465.8876953, 6395.4335938, -13442.8330078, 14205.3437500
3: -2650.5458984, 6771.7993164, -3601.2419434, 9098.9501953, -11749.4960938, 10373.0410156
4: -7830.4267578, 4702.0278320, -10513.1699219, 6345.6684570, -14176.0957031, 15215.1972656

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6881205, upper bound: 9157.3358701
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1161035, upper bound: 9160.5835272
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1161035, upper bound: 9160.5856011
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6166.5722656, 4599.8256836, -7320.9580078, 5438.5839844, -11605.1562500, 11920.7832031
1: -4904.3017578, 4425.1694336, -5813.3256836, 5230.9848633, -10135.2851562, 10238.4951172
2: -7120.4892578, 4803.7558594, -8435.4335938, 5710.5214844, -12831.0107422, 13239.1894531
3: -2674.7241211, 6859.6616211, -3207.7148438, 8108.7573242, -10783.4814453, 10067.3769531
4: -7905.9101562, 4758.4199219, -9365.0273438, 5660.8549805, -13566.7656250, 14123.4472656

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.5674413, upper bound: 9158.1497292
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6583.9965820, 4904.4194336, -8055.5351562, 5969.7250977, -12553.7187500, 12959.9550781
1: -5233.0166016, 4718.8378906, -6395.4716797, 5740.8032227, -10973.8203125, 11114.3076172
2: -7596.3549805, 5119.8242188, -9273.0371094, 6262.6469727, -13859.0009766, 14392.8613281
3: -2850.9125977, 7320.0991211, -3525.2802734, 8911.0253906, -11761.9375000, 10845.3779297
4: -8438.0537109, 5070.2192383, -10297.7148438, 6213.7075195, -14651.7597656, 15367.9335938

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.5629442, upper bound: 9157.3356230
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7668.4628906, 5695.8222656, -5217.9843750, 3886.9826660, -11555.4433594, 10913.8066406
1: -6089.5400391, 5476.5043945, -4150.7026367, 3735.8288574, -9825.3671875, 9627.2070312
2: -8832.9228516, 5977.6767578, -6032.2807617, 4068.1967773, -12901.1191406, 12009.9570312
3: -3366.0034180, 8492.4472656, -2273.4560547, 5791.7143555, -9157.7177734, 10765.9013672
4: -9808.8798828, 5931.0073242, -6699.9580078, 4032.4104004, -13841.2900391, 12630.9628906

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.1773301, upper bound: 9157.4580396
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3271465, upper bound: 9157.4640666
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7668.4628906, 5695.8222656, -6996.7739258, 5173.3305664, -12825.4580078, 12688.8105469
1: -6089.5400391, 5476.5043945, -5553.3051758, 4979.4038086, -11059.0048828, 11029.8095703
2: -8832.9228516, 5977.6767578, -8075.0971680, 5409.6850586, -14238.1230469, 14044.5166016
3: -3366.0034180, 8492.4472656, -3003.9746094, 7747.2231445, -11108.8476562, 11496.4218750
4: -9808.8798828, 5931.0073242, -8974.1601562, 5351.5878906, -15160.4667969, 14886.9150391

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.1773301, upper bound: 9158.1420344
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3271465, upper bound: 9158.1480402
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9350.4433594, 6922.4003906, -5217.9843750, 3886.9826660, -13182.4589844, 12140.3818359
1: -7414.0415039, 6663.9531250, -4150.7026367, 3735.8288574, -11121.2265625, 10814.6562500
2: -10767.2695312, 7265.3481445, -6032.2807617, 4068.1967773, -14783.9521484, 13297.6289062
3: -4067.5166016, 10346.7041016, -2273.4560547, 5791.7143555, -9859.2304688, 12608.6376953
4: -11960.8769531, 7191.9321289, -6699.9580078, 4032.4104004, -15931.3310547, 13891.8906250

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138742, upper bound: 9159.7132641
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138742, upper bound: 9159.7132641
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9350.4433594, 6922.4003906, -6996.7739258, 5173.3305664, -14442.8056641, 13888.5458984
1: -7414.0415039, 6663.9531250, -5553.3051758, 4979.4038086, -12344.3681641, 12202.2226562
2: -10767.2695312, 7265.3481445, -8075.0971680, 5409.6850586, -16098.4414062, 15298.0283203
3: -4067.5166016, 10346.7041016, -3003.9746094, 7747.2231445, -11788.1787109, 13322.8623047
4: -11960.8769531, 7191.9321289, -8974.1601562, 5351.5878906, -17223.1425781, 16115.0205078

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7484951
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7484951
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8160.6416016, 6048.0356445, -5977.5322266, 4447.2724609, -12607.9140625, 12025.5673828
1: -6479.2739258, 5815.7602539, -4751.3974609, 4275.8808594, -10755.1542969, 10567.1552734
2: -9395.2343750, 6342.2177734, -6898.5854492, 4651.2861328, -14046.5205078, 13240.8017578
3: -3569.1938477, 9031.1269531, -2605.6176758, 6632.8818359, -10202.0761719, 11636.7441406
4: -10435.3886719, 6291.0380859, -7665.0571289, 4615.4052734, -15050.7929688, 13956.0957031

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.1757828, upper bound: 9157.2192097
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3239837, upper bound: 9157.2252815
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8160.6416016, 6048.0356445, -7690.6572266, 5692.0659180, -13841.7333984, 13737.6191406
1: -6479.2739258, 5815.7602539, -6102.9794922, 5478.1386719, -11951.0468750, 11918.7402344
2: -9395.2343750, 6342.2177734, -8871.2363281, 5949.0122070, -15344.2460938, 15204.5322266
3: -3569.1938477, 9031.1269531, -3315.6516113, 8517.5019531, -12083.1894531, 12346.7783203
4: -10435.3886719, 6291.0380859, -9860.8076172, 5891.9072266, -16327.2958984, 16133.9853516

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.1757828, upper bound: 9157.3283729
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3239837, upper bound: 9157.3344334
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9807.6503906, 7251.0117188, -5977.5322266, 4447.2724609, -14204.3867188, 13228.5439453
1: -7776.5473633, 6979.5214844, -4751.3974609, 4275.8808594, -12026.5332031, 11730.9189453
2: -11290.4785156, 7604.8266602, -6898.5854492, 4651.2861328, -15896.6445312, 14503.4111328
3: -4256.6762695, 10846.8496094, -2605.6176758, 6632.8818359, -10889.5576172, 13443.0244141
4: -12543.2353516, 7529.3774414, -7665.0571289, 4615.4052734, -17103.6308594, 15194.4335938

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127391, upper bound: 9159.6798520
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127391, upper bound: 9159.6797398
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9807.6503906, 7251.0117188, -7690.6572266, 5692.0659180, -15424.0058594, 14914.7851562
1: -7776.5473633, 6979.5214844, -6102.9794922, 5478.1386719, -13209.2148438, 13069.5244141
2: -11290.4785156, 7604.8266602, -8871.2363281, 5949.0122070, -17166.6933594, 16434.4726562
3: -4256.6762695, 10846.8496094, -3315.6516113, 8517.5019531, -12749.4091797, 14137.8867188
4: -12543.2353516, 7529.3774414, -9860.8076172, 5891.9072266, -18354.2187500, 17339.8398438

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.7118562
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.7118562
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.60 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.2858128, upper bound: 9159.0587247
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9162.3993583, upper bound: 9160.3869796
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7120745
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7148390
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.6802222
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.6802222
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7099159
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7099159
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.6737074, upper bound: 9158.3550153
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9162.1768476, upper bound: 9160.3844795
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9161.1161035, upper bound: 9160.5835272
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9161.1161035, upper bound: 9160.5856011
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9160.6784706, upper bound: 9160.5820856
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9156.1773301, upper bound: 9157.4580396
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9157.3271465, upper bound: 9157.4640666
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9156.1773301, upper bound: 9158.1420344
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9157.3271465, upper bound: 9158.1480402
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7138742, upper bound: 9159.7132641
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7138742, upper bound: 9159.7132641
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7484951
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7138748, upper bound: 9159.7484951
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9156.1757828, upper bound: 9157.2192097
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9157.3239837, upper bound: 9157.2252815
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9156.1757828, upper bound: 9157.3283729
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.60
Output dim: 0, lower bound: -9157.3239837, upper bound: 9157.3344334
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7127391, upper bound: 9159.6798520
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7127391, upper bound: 9159.6797398
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.7118562
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.60
Output dim: 0, lower bound: -9159.7127397, upper bound: 9159.7118562

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5403.5800781, 4035.0004883, -5218.1777344, 3892.2780762, -9295.8574219, 9253.1777344
1: -4297.0322266, 3877.4035645, -4151.2275391, 3740.6689453, -8037.7006836, 8028.6308594
2: -6244.9731445, 4222.3583984, -6034.6562500, 4075.4570312, -10320.4296875, 10257.0126953
3: -2363.6596680, 6000.7211914, -2276.0891113, 5792.7021484, -8156.3613281, 8276.8105469
4: -6939.5683594, 4187.8696289, -6703.1074219, 4036.0100098, -10975.5781250, 10890.9765625

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2858128, upper bound: 9159.0587247
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2858128, upper bound: 9159.0587247
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5499.5751953, 4086.8627930, -5296.0888672, 3939.3850098, -9438.9570312, 9382.9511719
1: -4372.7910156, 3928.0920410, -4212.5874023, 3786.4443359, -8159.2348633, 8140.6782227
2: -6355.5400391, 4276.0205078, -6123.6137695, 4123.4516602, -10478.9892578, 10399.6347656
3: -2391.8969727, 6095.8769531, -2302.2666016, 5874.4990234, -8266.3945312, 8398.1416016
4: -7059.3813477, 4239.8994141, -6801.1391602, 4085.3779297, -11144.7548828, 11041.0351562

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0076712, upper bound: 9159.8205942
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8275483, upper bound: 9158.2389782
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -6116.9072266, 4546.7216797, -9886.1845703, 10090.2275391
1: -4247.0512695, 3818.8703613, -4861.9399414, 4371.2749023, -8618.3261719, 8680.8095703
2: -6172.7690430, 4157.7558594, -7060.9272461, 4751.9179688, -10924.6875000, 11218.6835938
3: -2321.1672363, 5922.5683594, -2658.0998535, 6786.5708008, -9107.7382812, 8580.6679688
4: -6855.8618164, 4119.5634766, -7845.5717773, 4714.1811523, -11570.0429688, 11965.1347656

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7120745
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7120743
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -6116.9072266, 4546.7216797, -10603.7128906, 10617.4267578
1: -4814.2382812, 4326.6323242, -4861.9399414, 4371.2749023, -9185.5126953, 9188.5722656
2: -6991.6362305, 4702.5898438, -7060.9272461, 4751.9179688, -11743.5546875, 11763.5166016
3: -2630.3791504, 6718.2563477, -2658.0998535, 6786.5708008, -9416.9501953, 9376.3564453
4: -7768.4980469, 4665.7036133, -7845.5717773, 4714.1811523, -12482.6796875, 12511.2744141

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7148390
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1463529, upper bound: 9160.7148390
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -5284.9296875, 3929.3693848, -9777.8779297, 9642.6240234
1: -4652.5683594, 4191.5620117, -4203.6108398, 3776.9763184, -8429.5419922, 8395.1699219
2: -6754.8217773, 4550.6113281, -6109.9667969, 4110.3730469, -10865.1953125, 10660.5781250
3: -2527.9526367, 6502.3999023, -2292.3381348, 5862.0122070, -8389.9648438, 8794.7382812
4: -7498.1806641, 4504.6772461, -6785.8813477, 4072.1474609, -11570.3271484, 11290.5585938

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6804305, upper bound: 9160.6957119
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6804305, upper bound: 9160.6957119
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -5284.9296875, 3929.3693848, -10464.2421875, 10153.8291016
1: -5194.1127930, 4684.6015625, -4203.6108398, 3776.9763184, -8971.0859375, 8888.2109375
2: -7539.5957031, 5082.8535156, -6109.9667969, 4110.3730469, -11649.9687500, 11192.8203125
3: -2830.4941406, 7265.7309570, -2292.3381348, 5862.0122070, -8692.5058594, 9558.0683594
4: -8374.8789062, 5033.6191406, -6785.8813477, 4072.1474609, -12447.0263672, 11819.5000000

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6804305, upper bound: 9160.6957119
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6804305, upper bound: 9160.6957119
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -5982.1625977, 4447.2451172, -10295.7519531, 10339.8574219
1: -4652.5683594, 4191.5620117, -4754.7382812, 4276.6386719, -8929.2070312, 8946.2988281
2: -6754.8217773, 4550.6113281, -6906.1962891, 4648.2451172, -11403.0664062, 11456.8076172
3: -2527.9526367, 6502.3999023, -2598.8793945, 6637.5014648, -9165.4541016, 9101.2792969
4: -7498.1806641, 4504.6772461, -7673.7441406, 4610.4365234, -12108.6152344, 12178.4218750

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7099157
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7099157
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -5982.1625977, 4447.2451172, -10982.1162109, 10851.0634766
1: -5194.1127930, 4684.6015625, -4754.7382812, 4276.6386719, -9470.7509766, 9439.3378906
2: -7539.5957031, 5082.8535156, -6906.1962891, 4648.2451172, -12187.8408203, 11989.0488281
3: -2830.4941406, 7265.7309570, -2598.8793945, 6637.5014648, -9467.9960938, 9864.6074219
4: -8374.8789062, 5033.6191406, -7673.7441406, 4610.4365234, -12985.3144531, 12707.3632812

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7096727
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6802222, upper bound: 9160.7096727
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5403.5800781, 4035.0004883, -7307.7363281, 5436.9794922, -10840.5595703, 11342.7363281
1: -4297.0322266, 3877.4035645, -5803.8715820, 5228.3720703, -9525.4042969, 9681.2744141
2: -6244.9731445, 4222.3583984, -8422.9111328, 5709.2612305, -11954.2324219, 12645.2685547
3: -2363.6596680, 6000.7211914, -3210.0629883, 8096.8330078, -10460.4912109, 9210.7841797
4: -6939.5683594, 4187.8696289, -9352.1855469, 5659.4228516, -12598.9912109, 13540.0537109

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9158.9478694, upper bound: 9157.2863533
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.2625106, upper bound: 9156.1714755
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5499.5751953, 4086.8627930, -7292.6503906, 5425.5839844, -10925.1572266, 11379.5136719
1: -4372.7910156, 3928.0920410, -5791.8022461, 5217.8437500, -9590.6347656, 9719.8945312
2: -6355.5400391, 4276.0205078, -8405.4501953, 5698.2836914, -12053.8232422, 12681.4707031
3: -2391.8969727, 6095.8769531, -3203.8239746, 8082.7675781, -10474.6640625, 9299.7001953
4: -7059.3813477, 4239.8994141, -9332.9580078, 5648.4838867, -12707.8652344, 13572.8574219

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4933450, upper bound: 9159.4302582
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3966401, upper bound: 9157.9530650
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5339.4643555, 3973.3225098, -8221.8310547, 6097.3745117, -11436.8388672, 12191.2529297
1: -4247.0512695, 3818.8703613, -6528.1015625, 5862.9794922, -10110.0312500, 10346.9716797
2: -6172.7690430, 4157.7558594, -9465.8876953, 6395.4335938, -12568.2011719, 13623.6435547
3: -2321.1672363, 5922.5683594, -3601.2419434, 9098.9501953, -11420.1171875, 9523.8105469
4: -6855.8618164, 4119.5634766, -10513.1699219, 6345.6684570, -13201.5302734, 14632.7333984

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1114309, upper bound: 9159.7201153
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1114309, upper bound: 9160.5835272
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6056.9916992, 4500.5209961, -8221.8310547, 6097.3745117, -12154.3662109, 12722.3515625
1: -4814.2382812, 4326.6323242, -6528.1015625, 5862.9794922, -10677.2177734, 10854.7343750
2: -6991.6362305, 4702.5898438, -9465.8876953, 6395.4335938, -13387.0703125, 14168.4765625
3: -2630.3791504, 6718.2563477, -3601.2419434, 9098.9501953, -11729.3291016, 10319.4980469
4: -7768.4980469, 4665.7036133, -10513.1699219, 6345.6684570, -14114.1660156, 15178.8710938

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1114309, upper bound: 9159.7222477
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1114309, upper bound: 9160.5856011
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -7320.9580078, 5438.5839844, -11287.0908203, 11678.6533203
1: -4652.5683594, 4191.5620117, -5813.3256836, 5230.9848633, -9883.5507812, 10004.8876953
2: -6754.8217773, 4550.6113281, -8435.4335938, 5710.5214844, -12465.3427734, 12986.0449219
3: -2527.9526367, 6502.3999023, -3207.7148438, 8108.7573242, -10636.7099609, 9710.1152344
4: -7498.1806641, 4504.6772461, -9365.0273438, 5660.8549805, -13159.0351562, 13869.7050781

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
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
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6732163, upper bound: 9159.6745598
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6732163, upper bound: 9160.6772072
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -7320.9580078, 5438.5839844, -11973.4560547, 12189.8593750
1: -5194.1127930, 4684.6015625, -5813.3256836, 5230.9848633, -10425.0947266, 10497.9277344
2: -7539.5957031, 5082.8535156, -8435.4335938, 5710.5214844, -13250.1171875, 13518.2871094
3: -2830.4941406, 7265.7309570, -3207.7148438, 8108.7573242, -10939.2519531, 10473.4443359
4: -8374.8789062, 5033.6191406, -9365.0273438, 5660.8549805, -14035.7343750, 14398.6464844

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
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
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6732163, upper bound: 9159.6745598
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6732163, upper bound: 9160.6772071
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5848.5097656, 4357.6953125, -8055.5351562, 5969.7250977, -11818.2314453, 12404.0869141
1: -4652.5683594, 4191.5620117, -6395.4716797, 5740.8032227, -10393.3710938, 10584.3906250
2: -6754.8217773, 4550.6113281, -9273.0371094, 6262.6469727, -13017.4677734, 13823.6484375
3: -2527.9526367, 6502.3999023, -3525.2802734, 8911.0253906, -11438.9785156, 10027.6796875
4: -7498.1806641, 4504.6772461, -10297.7148438, 6213.7075195, -13711.8876953, 14802.3925781

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6728911, upper bound: 9159.7196659
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6728911, upper bound: 9160.5812388
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6534.8730469, 4868.9013672, -8055.5351562, 5969.7250977, -12504.5957031, 12924.4365234
1: -5194.1127930, 4684.6015625, -6395.4716797, 5740.8032227, -10934.9160156, 11080.0712891
2: -7539.5957031, 5082.8535156, -9273.0371094, 6262.6469727, -13802.2421875, 14355.8906250
3: -2830.4941406, 7265.7309570, -3525.2802734, 8911.0253906, -11741.5195312, 10791.0078125
4: -8374.8789062, 5033.6191406, -10297.7148438, 6213.7075195, -14588.5859375, 15331.3339844

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6728911, upper bound: 9159.7190572
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6728911, upper bound: 9160.5812388
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9111.2812500, 6745.2363281, -5217.9843750, 3886.9826660, -12947.3105469, 11963.2207031
1: -7223.8691406, 6496.3237305, -4150.7026367, 3735.8288574, -10934.0429688, 10647.0263672
2: -10492.9638672, 7083.6816406, -6032.2807617, 4068.1967773, -14514.4599609, 13115.9619141
3: -3958.5959473, 10083.4667969, -2273.4560547, 5791.7143555, -9750.3105469, 12348.0244141
4: -11655.1445312, 7004.7363281, -6699.9580078, 4032.4104004, -15631.4599609, 13704.6943359

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3292509, upper bound: 9155.9161256
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3364198, upper bound: 9157.5061389
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9759.4541016, 7216.1552734, -5217.9843750, 3886.9826660, -13578.0820312, 12434.1376953
1: -7738.3730469, 6945.9658203, -4150.7026367, 3735.8288574, -11436.1386719, 11096.6679688
2: -11235.0390625, 7568.4702148, -6032.2807617, 4068.1967773, -15236.0849609, 13600.7509766
3: -4236.8867188, 10793.7646484, -2273.4560547, 5791.7143555, -10028.6015625, 13046.2773438
4: -12481.6035156, 7493.5170898, -6699.9580078, 4032.4104004, -16433.5273438, 14193.4736328

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3292509, upper bound: 9155.9161256
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3364198, upper bound: 9157.5061389
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9111.2812500, 6745.2363281, -6996.7739258, 5173.3305664, -14207.6562500, 13711.1005859
1: -7223.8691406, 6496.3237305, -5553.3051758, 4979.4038086, -12157.1855469, 12034.0068359
2: -10492.9638672, 7083.6816406, -8075.0971680, 5409.6850586, -15828.9492188, 15114.1425781
3: -3958.5959473, 10083.4667969, -3003.9746094, 7747.2231445, -11677.6425781, 13062.2480469
4: -11655.1445312, 7004.7363281, -8974.1601562, 5351.5878906, -16923.2734375, 15924.8876953

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6934120, upper bound: 9159.7484509
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6958459, upper bound: 9159.7462654
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9759.4541016, 7216.1552734, -6996.7739258, 5173.3305664, -14838.4277344, 14179.2695312
1: -7738.3730469, 6945.9658203, -5553.3051758, 4979.4038086, -12659.2812500, 12481.7011719
2: -11235.0390625, 7568.4702148, -8075.0971680, 5409.6850586, -16550.5722656, 15597.3916016
3: -4236.8867188, 10793.7646484, -3003.9746094, 7747.2231445, -11953.6162109, 13760.5009766
4: -12481.6035156, 7493.5170898, -8974.1601562, 5351.5878906, -17725.3417969, 16412.5722656

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6934120, upper bound: 9159.7484509
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6958459, upper bound: 9159.7462654
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9111.2812500, 6745.2363281, -5977.5322266, 4447.2724609, -13506.3759766, 12722.7685547
1: -7223.8691406, 6496.3237305, -4751.3974609, 4275.8808594, -11472.8886719, 11247.7197266
2: -10492.9638672, 7083.6816406, -6898.5854492, 4651.2861328, -15098.2929688, 13982.2675781
3: -3958.5959473, 10083.4667969, -2605.6176758, 6632.8818359, -10588.9951172, 12678.8544922
4: -11655.1445312, 7004.7363281, -7665.0571289, 4615.4052734, -16214.5302734, 14669.7929688

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7134064, upper bound: 9159.8590784
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7134066, upper bound: 9159.8590784
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9759.4541016, 7216.1552734, -5977.5322266, 4447.2724609, -14156.6953125, 13193.6875000
1: -7738.3730469, 6945.9658203, -4751.3974609, 4275.8808594, -11988.7138672, 11697.3613281
2: -11235.0390625, 7568.4702148, -6898.5854492, 4651.2861328, -15841.7148438, 14467.0546875
3: -4236.8867188, 10793.7646484, -2605.6176758, 6632.8818359, -10869.7685547, 13390.0166016
4: -12481.6035156, 7493.5170898, -7665.0571289, 4615.4052734, -17042.6093750, 15158.5732422

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7134064, upper bound: 9159.8586733
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.7134066, upper bound: 9159.8586733
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9111.2812500, 6745.2363281, -7690.6572266, 5692.0659180, -14724.1787109, 14387.9052734
1: -7223.8691406, 6496.3237305, -6102.9794922, 5478.1386719, -12654.1240234, 12571.5029297
2: -10492.9638672, 7083.6816406, -8871.2363281, 5949.0122070, -16368.3496094, 15889.6152344
3: -3958.5959473, 10083.4667969, -3315.6516113, 8517.5019531, -12435.6630859, 13372.0341797
4: -11655.1445312, 7004.7363281, -9860.8076172, 5891.9072266, -17465.1191406, 16787.0976562

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6934120, upper bound: 9159.7117586
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6945767, upper bound: 9159.6942405
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9759.4541016, 7216.1552734, -7690.6572266, 5692.0659180, -15376.3154297, 14879.7050781
1: -7738.3730469, 6945.9658203, -6102.9794922, 5478.1386719, -13171.3955078, 13035.7382812
2: -11235.0390625, 7568.4702148, -8871.2363281, 5949.0122070, -17111.7617188, 16397.9003906
3: -4236.8867188, 10793.7646484, -3315.6516113, 8517.5019531, -12729.5029297, 14084.8789062
4: -12481.6035156, 7493.5170898, -9860.8076172, 5891.9072266, -18293.1972656, 17303.7226562

Time for backsubstitution: 1.39 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=11463.63671875
rel_dist={0: [-9164.405028251073, 9164.405028251073]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1121.88 seconds
