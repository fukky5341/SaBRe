## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 4711.385590072957


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836)
1: (-239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813)
2: (-164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602)
3: (-201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642)
4: (-160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.13 + 1.90 = 5.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4711.4327044, upper bound: 4711.4327043

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326994
time: 0.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4327042
time: 0.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.36 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326994
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4327042

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2659.2883301, 2176.5551758, -2804.4550781, 2266.1918945, -4925.4799805, 4981.0102539
1: -223.4974060, 158.1744995, -233.4198761, 166.2590485, -389.7564087, 391.5943604
2: -152.4617615, 258.6788635, -160.0506287, 270.1564941, -422.6182556, 418.7294006
3: -187.7437286, 379.7648010, -196.8087921, 397.9906616, -585.7343750, 576.5736084
4: -148.2258453, 262.6078796, -155.8397217, 273.6827087, -421.9085388, 418.4475708

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4326994
time: 0.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326995, upper bound: 4711.4326995
time: 0.57 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4262.5400391, 3428.3508301, -2874.1862793, 2310.3144531, -6572.8544922, 6302.5371094
1: -353.8169861, 251.2803650, -238.2370300, 170.1911163, -524.0081177, 489.5173035
2: -240.3132782, 408.1385803, -163.6802368, 275.7773743, -516.0905762, 571.8188477
3: -296.6077576, 603.9949341, -201.1240082, 406.8107605, -703.4185181, 805.1188354
4: -236.0597076, 413.7591553, -159.4575500, 279.1725159, -515.2321777, 573.2166748

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327042
time: 0.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327042
time: 0.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.58 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4326994
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 0, lower bound: -4711.4326995, upper bound: 4711.4326995
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327042
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327042

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2659.2883301, 2176.5551758, -2659.2883301, 2176.5551758, -4835.8437500, 4835.8437500
1: -223.4974060, 158.1744995, -223.4974060, 158.1744995, -381.6719055, 381.6719055
2: -152.4617615, 258.6788635, -152.4617615, 258.6788635, -411.1405945, 411.1405945
3: -187.7437286, 379.7648010, -187.7437286, 379.7648010, -567.5085449, 567.5085449
4: -148.2258453, 262.6078796, -148.2258453, 262.6078796, -410.8336792, 410.8336792

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326993, upper bound: 4711.4326952
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4326995
time: 0.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2659.2883301, 2176.5551758, -4262.5400391, 3428.3508301, -6087.6391602, 6439.0952148
1: -223.4974060, 158.1744995, -353.8169861, 251.2803650, -474.7776489, 511.9914856
2: -152.4617615, 258.6788635, -240.3132782, 408.1385803, -560.6003418, 498.9920044
3: -187.7437286, 379.7648010, -296.6077576, 603.9949341, -791.7386475, 676.3725586
4: -148.2258453, 262.6078796, -236.0597076, 413.7591553, -561.9849854, 498.6675110

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326993, upper bound: 4711.4326952
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4326995
time: 0.65 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4262.5400391, 3428.3508301, -2659.2883301, 2176.5551758, -6439.0952148, 6087.6391602
1: -353.8169861, 251.2803650, -223.4974060, 158.1744995, -511.9914856, 474.7776489
2: -240.3132782, 408.1385803, -152.4617615, 258.6788635, -498.9920044, 560.6003418
3: -296.6077576, 603.9949341, -187.7437286, 379.7648010, -676.3725586, 791.7386475
4: -236.0597076, 413.7591553, -148.2258453, 262.6078796, -498.6675110, 561.9849854

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326995, upper bound: 4711.4327020
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326995, upper bound: 4711.4327042
time: 0.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4262.5400391, 3428.3508301, -4262.5400391, 3428.3508301, -7676.7832031, 7676.7832031
1: -353.8169861, 251.2803650, -353.8169861, 251.2803650, -604.0053711, 604.0053711
2: -240.3132782, 408.1385803, -240.3132782, 408.1385803, -647.9954224, 647.9954224
3: -296.6077576, 603.9949341, -296.6077576, 603.9949341, -899.8701172, 899.8701172
4: -236.0597076, 413.7591553, -236.0597076, 413.7591553, -648.7472534, 648.7472534

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327020
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327042
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.06 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326993, upper bound: 4711.4326952
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4326995
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326993, upper bound: 4711.4326952
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4326995
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326995, upper bound: 4711.4327020
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326995, upper bound: 4711.4327042
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327020
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 0, lower bound: -4711.4326996, upper bound: 4711.4327042

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2474.2604980, 2026.2491455, -2622.6450195, 2146.7817383, -4621.0419922, 4648.8935547
1: -207.8417511, 147.3905945, -220.4122925, 156.0309296, -363.8725281, 367.8028259
2: -142.0680542, 241.4571381, -150.4132843, 255.2879028, -397.3559570, 391.8704224
3: -174.8747406, 354.3904114, -185.2120361, 374.7150269, -549.5896606, 539.6024170
4: -138.1492004, 244.4403076, -146.2180023, 259.0063782, -397.1555481, 390.6583252

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326481, upper bound: 4711.4326628
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326994, upper bound: 4711.4326946
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2640.7280273, 2163.3662109, -2659.2883301, 2176.5551758, -4817.2832031, 4822.6542969
1: -222.0673218, 157.1172638, -223.4974060, 158.1744995, -380.2418213, 380.6146851
2: -151.4788208, 256.9772034, -152.4617615, 258.6788635, -410.1576538, 409.4389648
3: -186.5455170, 377.1846619, -187.7437286, 379.7648010, -566.3102417, 564.9284058
4: -147.1840057, 261.0017700, -148.2258453, 262.6078796, -409.7918091, 409.2276001

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326952, upper bound: 4711.4326994
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326952, upper bound: 4711.4326995
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2474.2604980, 2026.2491455, -4225.2778320, 3398.8105469, -5873.0708008, 6251.5268555
1: -207.8417511, 147.3905945, -350.6881714, 249.1457214, -456.9874573, 498.0787048
2: -142.0680542, 241.4571381, -238.3107300, 404.7239380, -546.7918091, 479.7678223
3: -174.8747406, 354.3904114, -294.1302490, 598.9885254, -773.8632812, 648.5205078
4: -138.1492004, 244.4403076, -234.1088409, 410.1802063, -548.3293457, 478.5491333

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326820, upper bound: 4711.4326952
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326952
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2640.7280273, 2163.3662109, -4262.5400391, 3428.3508301, -6069.0786133, 6425.9062500
1: -222.0673218, 157.1172638, -353.8169861, 251.2803650, -473.3476257, 510.9342651
2: -151.4788208, 256.9772034, -240.3132782, 408.1385803, -559.6174316, 497.2904358
3: -186.5455170, 377.1846619, -296.6077576, 603.9949341, -790.5404053, 673.7924194
4: -147.1840057, 261.0017700, -236.0597076, 413.7591553, -560.9431763, 497.0614624

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326995
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326994
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4238.5522461, 3413.1520996, -2656.1901855, 2174.3442383, -6412.8964844, 6069.3417969
1: -352.1457825, 250.0003510, -223.2601013, 157.9997406, -510.1455078, 473.2603149
2: -239.1118622, 406.2689819, -152.2924347, 258.4067688, -497.5186157, 558.5614014
3: -294.9869385, 600.7194214, -187.5287018, 379.3493042, -674.3362427, 788.2480469
4: -234.7031097, 411.9863281, -148.0530853, 262.3397217, -497.0427856, 560.0393677

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4327020
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4327019
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4225.1215820, 3397.4221191, -2659.2883301, 2176.5551758, -6401.6767578, 6056.7104492
1: -350.5905762, 249.0515442, -223.4974060, 158.1744995, -508.7650757, 472.5488892
2: -238.1571045, 404.5286255, -152.4617615, 258.6788635, -496.8359070, 556.9903564
3: -293.9518738, 598.6807251, -187.7437286, 379.7648010, -673.7166748, 786.4244385
4: -233.9711151, 410.0380859, -148.2258453, 262.6078796, -496.5789490, 558.2639160

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4327020
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326994, upper bound: 4711.4327042
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4238.5522461, 3413.1520996, -4259.5009766, 3426.2268066, -7648.4702148, 7655.9658203
1: -352.1457825, 250.0003510, -353.5942993, 251.1073608, -601.9461670, 602.2888184
2: -239.1118622, 406.2689819, -240.1546936, 407.8804626, -645.9863892, 645.6536255
3: -294.9869385, 600.7194214, -296.4073181, 603.5930786, -897.1657715, 895.9796753
4: -234.7031097, 411.9863281, -235.8995972, 413.5035400, -646.6525269, 646.3869019

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4327019
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327019, upper bound: 4711.4327019
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4225.1215820, 3397.4221191, -4262.5400391, 3428.3508301, -7639.6123047, 7645.9633789
1: -350.5905762, 249.0515442, -353.8169861, 251.2803650, -600.7966919, 601.8003540
2: -238.1571045, 404.5286255, -240.3132782, 408.1385803, -645.8991699, 644.3979492
3: -293.9518738, 598.6807251, -296.6077576, 603.9949341, -897.2773438, 894.5700684
4: -233.9711151, 410.0380859, -236.0597076, 413.7591553, -646.7045898, 645.0611572

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327019, upper bound: 4711.4327042
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327018, upper bound: 4711.4327042
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.50 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326481, upper bound: 4711.4326628
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326994, upper bound: 4711.4326946
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326952, upper bound: 4711.4326994
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326952, upper bound: 4711.4326995
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326820, upper bound: 4711.4326952
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326952
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326995
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326994
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4327020
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4327019
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4327020
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4326994, upper bound: 4711.4327042
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4327019
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327019, upper bound: 4711.4327019
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327019, upper bound: 4711.4327042
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 0, lower bound: -4711.4327018, upper bound: 4711.4327042

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2471.1867676, 2023.9276123, -2586.8425293, 2117.5969238, -4588.7836914, 4610.7695312
1: -207.5978851, 147.2133636, -217.2331390, 154.0602570, -361.6581421, 364.4465027
2: -141.8961792, 241.1782532, -148.2919769, 251.8397980, -393.7359009, 389.4702148
3: -174.6617432, 353.9673767, -182.3947144, 369.3445435, -544.0062866, 536.3620605
4: -137.9804077, 244.1592407, -143.9947968, 255.5217285, -393.5021057, 388.1540527

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326302, upper bound: 4711.4326510
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326303, upper bound: 4711.4326434
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2474.2604980, 2026.2491455, -2586.8208008, 2116.0593262, -4590.3198242, 4613.0698242
1: -207.8417511, 147.3905945, -217.2869110, 153.8699646, -361.7116089, 364.6773987
2: -142.0680542, 241.4571381, -148.2860413, 251.7484131, -393.8164368, 389.7431030
3: -174.8747406, 354.3904114, -182.6374817, 369.5114441, -544.3861694, 537.0277710
4: -138.1492004, 244.4403076, -144.1833191, 255.3173218, -393.4664917, 388.6236267

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326796, upper bound: 4711.4326681
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326993, upper bound: 4711.4326946
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2640.7280273, 2163.3662109, -2474.2604980, 2026.2491455, -4666.9770508, 4637.6269531
1: -222.0673218, 157.1172638, -207.8417511, 147.3905945, -369.4578552, 364.9590149
2: -151.4788208, 256.9772034, -142.0680542, 241.4571381, -392.9359741, 399.0452271
3: -186.5455170, 377.1846619, -174.8747406, 354.3904114, -540.9357910, 552.0593872
4: -147.1840057, 261.0017700, -138.1492004, 244.4403076, -391.6242981, 399.1509705

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326628, upper bound: 4711.4326481
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326993
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2640.7280273, 2163.3662109, -2640.7280273, 2163.3662109, -4804.0942383, 4804.0942383
1: -222.0673218, 157.1172638, -222.0673218, 157.1172638, -379.1845703, 379.1845703
2: -151.4788208, 256.9772034, -151.4788208, 256.9772034, -408.4560242, 408.4560242
3: -186.5455170, 377.1846619, -186.5455170, 377.1846619, -563.7300415, 563.7301025
4: -147.1840057, 261.0017700, -147.1840057, 261.0017700, -408.1857605, 408.1857605

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326628, upper bound: 4711.4326481
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326994
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2454.7463379, 2011.8746338, -3957.0104980, 3202.7155762, -5657.4619141, 5968.8852539
1: -206.3506165, 146.2787933, -330.2330017, 233.8417053, -440.1923218, 476.5117798
2: -141.0334625, 239.7319946, -224.7737122, 381.4388123, -522.4722900, 464.5057068
3: -173.5972748, 351.8324280, -277.2823486, 563.4902344, -737.0875244, 629.1146851
4: -137.1487274, 242.6903534, -220.5880585, 386.6083679, -523.7570801, 463.2784119

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326836
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326713, upper bound: 4711.4326813
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2474.2604980, 2026.2491455, -4221.2416992, 3396.1269531, -5870.3876953, 6247.4907227
1: -207.8417511, 147.3905945, -350.4003601, 248.9167480, -456.7584839, 497.7909546
2: -142.0680542, 241.4571381, -238.0987396, 404.3979187, -546.4658813, 479.5558472
3: -174.8747406, 354.3904114, -293.8757324, 598.4785767, -773.3533325, 648.2661133
4: -138.1492004, 244.4403076, -233.9052734, 409.8489990, -547.9980469, 478.3455811

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326975, upper bound: 4711.4326571
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326952
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2637.7470703, 2161.2309570, -4238.5522461, 3413.1520996, -6050.8994141, 6399.7832031
1: -221.8384857, 156.9491119, -352.1457825, 250.0003510, -471.8387451, 509.0948486
2: -151.3195648, 256.7155762, -239.1118622, 406.2689819, -557.5885620, 495.8274231
3: -186.3431854, 376.7842407, -294.9869385, 600.7194214, -787.0625610, 671.7711182
4: -147.0182190, 260.7435608, -234.7031097, 411.9863281, -559.0044556, 495.4466248

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326961
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326994
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2640.7280273, 2163.3662109, -4225.1215820, 3397.4221191, -6038.1499023, 6388.4877930
1: -222.0673218, 157.1172638, -350.5905762, 249.0515442, -471.1188660, 507.7078247
2: -151.4788208, 256.9772034, -238.1571045, 404.5286255, -556.0074463, 495.1343079
3: -186.5455170, 377.1846619, -293.9518738, 598.6807251, -785.2261963, 671.1365356
4: -147.1840057, 261.0017700, -233.9711151, 410.0380859, -557.2221069, 494.9729004

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326961
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326995
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4238.5522461, 3413.1520996, -2623.4931641, 2148.1828613, -6386.7353516, 6036.6455078
1: -352.1457825, 250.0003510, -220.4178467, 156.2333374, -508.3791199, 470.4181213
2: -239.1118622, 406.2689819, -150.4342041, 255.3027191, -494.4145813, 556.7031860
3: -294.9869385, 600.7194214, -185.0433960, 374.4764099, -669.4633789, 785.7628174
4: -234.7031097, 411.9863281, -146.0088806, 259.2086487, -493.9117432, 557.9952393

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326632
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326342
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4238.5522461, 3413.1520996, -2623.3503418, 2145.4111328, -6383.9633789, 6036.5024414
1: -352.1457825, 250.0003510, -220.3308563, 155.9863739, -508.1321411, 470.3312073
2: -239.1118622, 406.2689819, -150.3207397, 255.0989532, -494.2108154, 556.5897217
3: -294.9869385, 600.7194214, -185.1497040, 374.5212708, -669.5081787, 785.8691406
4: -234.7031097, 411.9863281, -146.1812134, 258.8603210, -493.5634155, 558.1674805

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326231, upper bound: 4711.4326653
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326344
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4188.3481445, 3368.0158691, -2474.2604980, 2026.2491455, -6214.5971680, 5842.2763672
1: -347.4891968, 246.9200592, -207.8417511, 147.3905945, -494.8797302, 454.7617798
2: -236.1671448, 401.1387634, -142.0680542, 241.4571381, -477.6242676, 543.2067261
3: -291.4927368, 593.7044067, -174.8747406, 354.3904114, -645.8831787, 768.5791626
4: -232.0213470, 406.4736633, -138.1492004, 244.4403076, -476.4615479, 544.6228638

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326812
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4327020
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4225.1215820, 3397.4221191, -2640.7280273, 2163.3662109, -6388.4877930, 6038.1499023
1: -350.5905762, 249.0515442, -222.0673218, 157.1172638, -507.7078247, 471.1188660
2: -238.1571045, 404.5286255, -151.4788208, 256.9772034, -495.1343079, 556.0074463
3: -293.9518738, 598.6807251, -186.5455170, 377.1846619, -671.1365356, 785.2261963
4: -233.9711151, 410.0380859, -147.1840057, 261.0017700, -494.9729004, 557.2221069

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326681, upper bound: 4711.4326848
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326676, upper bound: 4711.4326755
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4238.5522461, 3413.1520996, -4238.5522461, 3413.1520996, -7633.2006836, 7633.2006836
1: -352.1457825, 250.0003510, -352.1457825, 250.0003510, -600.6619263, 600.6619263
2: -239.1118622, 406.2689819, -239.1118622, 406.2689819, -644.1290894, 644.1290283
3: -294.9869385, 600.7194214, -294.9869385, 600.7194214, -893.9665527, 893.9665527
4: -234.7031097, 411.9863281, -234.7031097, 411.9863281, -644.7823486, 644.7823486

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326636
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4238.5522461, 3413.1520996, -4225.1215820, 3397.4221191, -7619.7792969, 7621.8271484
1: -352.1457825, 250.0003510, -350.5905762, 249.0515442, -599.9120483, 599.3054199
2: -239.1118622, 406.2689819, -238.1571045, 404.5286255, -642.6461182, 643.7172852
3: -294.9869385, 600.7194214, -293.9518738, 598.6807251, -892.2664795, 893.5850220
4: -234.7031097, 411.9863281, -233.9711151, 410.0380859, -643.2215576, 644.5026245

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326658
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4225.1215820, 3397.4221191, -4238.5522461, 3413.1520996, -7621.8271484, 7619.7792969
1: -350.5905762, 249.0515442, -352.1457825, 250.0003510, -599.3053589, 599.9121094
2: -238.1571045, 404.5286255, -239.1118622, 406.2689819, -643.7172852, 642.6460571
3: -293.9518738, 598.6807251, -294.9869385, 600.7194214, -893.5850220, 892.2664795
4: -233.9711151, 410.0380859, -234.7031097, 411.9863281, -644.5026245, 643.2215576

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326680
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326598
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4225.1215820, 3397.4221191, -4225.1215820, 3397.4221191, -7608.7919922, 7608.7919922
1: -350.5905762, 249.0515442, -350.5905762, 249.0515442, -598.5916748, 598.5917358
2: -238.1571045, 404.5286255, -238.1571045, 404.5286255, -642.3016968, 642.3016357
3: -293.9518738, 598.6807251, -293.9518738, 598.6807251, -891.9772949, 891.9772949
4: -233.9711151, 410.0380859, -233.9711151, 410.0380859, -643.0184326, 643.0184326

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326848
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326755
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.25 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326302, upper bound: 4711.4326510
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326303, upper bound: 4711.4326434
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326796, upper bound: 4711.4326681
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326993, upper bound: 4711.4326946
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326628, upper bound: 4711.4326481
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326993
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326628, upper bound: 4711.4326481
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326994
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326836
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326713, upper bound: 4711.4326813
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326975, upper bound: 4711.4326571
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326952
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326961
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4327020, upper bound: 4711.4326994
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326961
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4327042, upper bound: 4711.4326995
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326632
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326342
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326231, upper bound: 4711.4326653
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326344
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326812
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4327020
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326681, upper bound: 4711.4326848
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326676, upper bound: 4711.4326755
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326636
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326658
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326680
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326598
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326848
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.25
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326755

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2454.7104492, 2012.9052734, -2586.8425293, 2117.5969238, -4572.3076172, 4599.7470703
1: -206.4333344, 146.2502899, -217.2331390, 154.0602570, -360.4935913, 363.4834290
2: -141.0152283, 239.7950745, -148.2919769, 251.8397980, -392.8550110, 388.0870361
3: -173.6112213, 351.8038635, -182.3947144, 369.3445435, -542.9557495, 534.1984863
4: -137.1381531, 242.8017426, -143.9947968, 255.5217285, -392.6598816, 386.7965393

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326302, upper bound: 4711.4326434
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326302, upper bound: 4711.4326435
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2489.4003906, 2041.7442627, -2564.7705078, 2102.8024902, -4592.2031250, 4606.5146484
1: -209.4646606, 148.3458862, -215.6605988, 152.7794952, -362.2441406, 364.0064697
2: -143.4649353, 243.4570923, -147.1314697, 249.9651794, -393.4301147, 390.5885620
3: -176.5990448, 357.0031433, -180.9984283, 366.4758911, -543.0747681, 538.0015869
4: -139.6437836, 246.2940979, -142.8790588, 253.6940155, -393.3377380, 389.1731567

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326303, upper bound: 4711.4326433
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326303, upper bound: 4711.4326434
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2371.5834961, 1956.7266846, -2554.5983887, 2094.7089844, -4466.2919922, 4511.3242188
1: -200.5023193, 141.4376984, -215.0108032, 151.9981079, -352.5003662, 356.4484558
2: -136.6167145, 232.6844788, -146.5897827, 249.0157776, -385.6325073, 379.2742615
3: -168.3379211, 340.8910217, -180.5995026, 365.3050842, -533.6430054, 521.4905396
4: -132.8790741, 235.8761139, -142.5211792, 252.6774445, -385.5565186, 378.3972778

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326679
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326681
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2497.7451172, 2065.9453125, -2540.1643066, 2085.3125000, -4583.0576172, 4606.1083984
1: -211.6912079, 148.9949188, -213.9627838, 151.2615509, -362.9527588, 362.9577026
2: -144.2149658, 245.2859650, -145.9255981, 247.9947205, -392.2096252, 391.2115173
3: -177.5965576, 358.9290771, -179.7497864, 363.6231689, -541.2197266, 538.6787720
4: -140.0048370, 249.1302643, -141.8080597, 251.5580750, -391.5629272, 390.9383240

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326946
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326946
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -2471.1867676, 2023.9276123, -4632.1079102, 4608.7788086
1: -219.3154755, 155.3539429, -207.5978851, 147.2133636, -366.5288086, 362.9518433
2: -149.6423492, 253.9204102, -141.8961792, 241.1782532, -390.8206177, 395.8164978
3: -184.0760956, 372.3533936, -174.6617432, 353.9673767, -538.0434570, 547.0150146
4: -145.1245575, 257.9199524, -137.9804077, 244.1592407, -389.2838135, 395.9003296

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326508, upper bound: 4711.4326300
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326434, upper bound: 4711.4326302
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2604.8234863, 2132.1284180, -2474.2604980, 2026.2491455, -4631.0727539, 4606.3886719
1: -218.8876343, 154.9284515, -207.8417511, 147.3905945, -366.2781677, 362.7701721
2: -149.2814026, 253.3880920, -142.0680542, 241.4571381, -390.7385254, 395.4560547
3: -183.8647308, 371.9350586, -174.8747406, 354.3904114, -538.2551270, 546.8098145
4: -145.1332245, 257.2478333, -138.1492004, 244.4403076, -389.5735474, 395.3969421

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326681, upper bound: 4711.4326796
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326993
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -2637.7470703, 2161.2309570, -4769.4111328, 4775.3388672
1: -219.3154755, 155.3539429, -221.8384857, 156.9491119, -376.2644958, 377.1923523
2: -149.6423492, 253.9204102, -151.3195648, 256.7155762, -406.3579102, 405.2399597
3: -184.0760956, 372.3533936, -186.3431854, 376.7842407, -560.8603516, 558.6964111
4: -145.1245575, 257.9199524, -147.0182190, 260.7435608, -405.8681030, 404.9381104

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326961
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326961
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2604.8234863, 2132.1284180, -2640.7280273, 2163.3662109, -4768.1894531, 4772.8559570
1: -218.8876343, 154.9284515, -222.0673218, 157.1172638, -376.0048828, 376.9957581
2: -149.2814026, 253.3880920, -151.4788208, 256.9772034, -406.2586060, 404.8669128
3: -183.8647308, 371.9350586, -186.5455170, 377.1846619, -561.0493774, 558.4805298
4: -145.1332245, 257.2478333, -147.1840057, 261.0017700, -406.1350098, 404.4317322

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326973
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326995
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2326.9038086, 1911.5601807, -3957.0104980, 3202.7155762, -5529.6191406, 5868.5708008
1: -195.9856110, 138.7174988, -330.2330017, 233.8417053, -429.8272705, 468.9505005
2: -133.7745819, 227.4561615, -224.7737122, 381.4388123, -515.2133789, 452.2298584
3: -164.6847229, 333.4387207, -277.2823486, 563.4902344, -728.1748657, 610.7210083
4: -130.0346527, 230.5344849, -220.5880585, 386.6083679, -516.6430054, 451.1225586

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326713, upper bound: 4711.4326836
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326836
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2372.1406250, 1932.3253174, -3957.0104980, 3202.7155762, -5574.8554688, 5889.3359375
1: -198.3720856, 141.0652466, -330.2330017, 233.8417053, -432.2137451, 471.2981873
2: -135.6730499, 230.5015564, -224.7737122, 381.4388123, -517.1118774, 455.2752686
3: -167.0089569, 338.7012024, -277.2823486, 563.4902344, -730.4992065, 615.9834595
4: -131.9194641, 233.0548859, -220.5880585, 386.6083679, -518.5277710, 453.6429443

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326815
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326814
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2457.7775879, 2000.8471680, -4217.9741211, 3393.4877930, -5851.2656250, 6218.8208008
1: -205.4572144, 146.1996307, -350.1311951, 248.7195129, -454.1767273, 496.3307800
2: -140.6163330, 238.8777161, -237.9082947, 404.0860596, -544.7023315, 476.7859802
3: -173.0384827, 351.1656494, -293.6462097, 598.0184937, -771.0568848, 644.8118896
4: -136.7552795, 241.4511871, -233.7287292, 409.5313110, -546.2865601, 475.1798096

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326932, upper bound: 4711.4326572
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326930, upper bound: 4711.4326571
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2452.1994629, 2009.2750244, -4221.2416992, 3396.1269531, -5848.3256836, 6230.5166016
1: -206.0929565, 146.1060638, -350.4003601, 248.9167480, -455.0097046, 496.5064087
2: -140.8576050, 239.4400330, -238.0987396, 404.3979187, -545.2554932, 477.5387573
3: -173.3688965, 351.4163818, -293.8757324, 598.4785767, -771.8474731, 645.2921143
4: -136.9850616, 242.3912659, -233.9052734, 409.8489990, -546.8340454, 476.2965393

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326962, upper bound: 4711.4326952
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326963, upper bound: 4711.4326952
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -4238.5522461, 3413.1520996, -6021.3320312, 6376.1445312
1: -219.3154755, 155.3539429, -352.1457825, 250.0003510, -469.3157349, 507.4997253
2: -149.6423492, 253.9204102, -239.1118622, 406.2689819, -555.9113159, 493.0322571
3: -184.0760956, 372.3533936, -294.9869385, 600.7194214, -784.7955322, 667.3402100
4: -145.1245575, 257.9199524, -234.7031097, 411.9863281, -557.1108398, 492.6230469

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326632, upper bound: 4711.4326231
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326342, upper bound: 4711.4326232
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2604.8234863, 2132.1284180, -4238.5522461, 3413.1520996, -6017.9755859, 6370.6801758
1: -218.8876343, 154.9284515, -352.1457825, 250.0003510, -468.8879089, 507.0742188
2: -149.2814026, 253.3880920, -239.1118622, 406.2689819, -555.5502930, 492.4999390
3: -183.8647308, 371.9350586, -294.9869385, 600.7194214, -784.5841675, 666.9219971
4: -145.1332245, 257.2478333, -234.7031097, 411.9863281, -557.1194458, 491.9508667

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326632, upper bound: 4711.4326494
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326499
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -4225.1215820, 3397.4221191, -6005.6025391, 6362.7138672
1: -219.3154755, 155.3539429, -350.5905762, 249.0515442, -468.3669739, 505.9445190
2: -149.6423492, 253.9204102, -238.1571045, 404.5286255, -554.1708984, 492.0774841
3: -184.0760956, 372.3533936, -293.9518738, 598.6807251, -782.7568359, 666.3052368
4: -145.1245575, 257.9199524, -233.9711151, 410.0380859, -555.1626587, 491.8910522

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326631, upper bound: 4711.4326228
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326231
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2604.8234863, 2132.1284180, -4225.1215820, 3397.4221191, -6002.2456055, 6357.2500000
1: -218.8876343, 154.9284515, -350.5905762, 249.0515442, -467.9391479, 505.5190125
2: -149.2814026, 253.3880920, -238.1571045, 404.5286255, -553.8098755, 491.5451660
3: -183.8647308, 371.9350586, -293.9518738, 598.6807251, -782.5454712, 665.8869629
4: -145.1332245, 257.2478333, -233.9711151, 410.0380859, -555.1713257, 491.2189026

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326631, upper bound: 4711.4326681
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326681
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4223.2407227, 3403.4516602, -2623.4931641, 2148.1828613, -6371.4233398, 6026.9443359
1: -351.0947876, 249.1456909, -220.4178467, 156.2333374, -507.3281250, 469.5634766
2: -238.3108063, 405.0757751, -150.4342041, 255.3027191, -493.6134644, 555.5100098
3: -294.0272827, 598.7977905, -185.0433960, 374.4764099, -668.5036621, 783.8411865
4: -233.9154510, 410.7941895, -146.0088806, 259.2086487, -493.1240540, 556.8031006

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326086, upper bound: 4711.4326584
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326116, upper bound: 4711.4326525
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4298.6142578, 3449.6342773, -2615.6386719, 2142.8979492, -6441.5122070, 6065.2729492
1: -356.1275635, 253.2917786, -219.8586121, 155.7863617, -511.9139404, 473.1503906
2: -241.8614502, 411.0473633, -150.0320129, 254.6454926, -496.5068970, 561.0793457
3: -298.4373779, 608.0323486, -184.5565948, 373.4658813, -671.9031982, 792.5889282
4: -237.5672913, 416.4108276, -145.6148529, 258.5678711, -496.1351624, 562.0256958

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326086, upper bound: 4711.4326204
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326116, upper bound: 4711.4326204
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4223.2407227, 3403.4516602, -2623.3503418, 2145.4111328, -6368.6518555, 6026.8012695
1: -351.0947876, 249.1456909, -220.3308563, 155.9863739, -507.0811768, 469.4765320
2: -238.3108063, 405.0757751, -150.3207397, 255.0989532, -493.4097290, 555.3964844
3: -294.0272827, 598.7977905, -185.1497040, 374.5212708, -668.5484009, 783.9475098
4: -233.9154510, 410.7941895, -146.1812134, 258.8603210, -492.7757568, 556.9754028

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326114, upper bound: 4711.4326590
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326453, upper bound: 4711.4326607
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4298.6142578, 3449.6342773, -2615.6457520, 2140.0869141, -6438.7011719, 6065.2802734
1: -356.1275635, 253.2917786, -219.7704620, 155.5427246, -511.6702881, 473.0622253
2: -241.8614502, 411.0473633, -149.9186859, 254.4408722, -496.3023071, 560.9660034
3: -298.4373779, 608.0323486, -184.6638794, 373.5140381, -671.9512329, 792.6962280
4: -237.5672913, 416.4108276, -145.7918549, 258.2154541, -495.7827454, 562.2026978

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326118, upper bound: 4711.4326204
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326459, upper bound: 4711.4326205
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3928.9365234, 3179.2512207, -2454.7463379, 2011.8746338, -5940.8110352, 5633.9975586
1: -327.7936096, 232.1559448, -206.3506165, 146.2787933, -474.0723877, 438.5065613
2: -223.1448822, 378.7071838, -141.0334625, 239.7319946, -462.8768921, 519.7406616
3: -275.2762756, 559.4517212, -173.5972748, 351.8324280, -627.1087036, 733.0490112
4: -219.0125580, 383.7878418, -137.1487274, 242.6903534, -461.7029114, 520.9365845

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326828, upper bound: 4711.4326707
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326808, upper bound: 4711.4326707
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4184.3520508, 3365.3505859, -2474.2604980, 2026.2491455, -6210.6010742, 5839.6113281
1: -347.2036133, 246.6929321, -207.8417511, 147.3905945, -494.5941467, 454.5345764
2: -235.9562225, 400.8153687, -142.0680542, 241.4571381, -477.4133606, 542.8833008
3: -291.2399597, 593.1980591, -174.8747406, 354.3904114, -645.6301880, 768.0728149
4: -231.8189392, 406.1447144, -138.1492004, 244.4403076, -476.2592468, 544.2938232

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326570, upper bound: 4711.4326978
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4327016
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4209.2080078, 3387.2622070, -2640.7280273, 2163.3662109, -6372.5742188, 6027.9897461
1: -349.4930725, 248.1587067, -222.0673218, 157.1172638, -506.6103516, 470.2260132
2: -237.3243256, 403.2819519, -151.4788208, 256.9772034, -494.3015137, 554.7607422
3: -292.9544067, 596.6747437, -186.5455170, 377.1846619, -670.1390381, 783.2202148
4: -233.1558380, 408.7913513, -147.1840057, 261.0017700, -494.1575928, 555.9753418

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326680
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326227, upper bound: 4711.4326849
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4291.9921875, 3437.1938477, -2632.8356934, 2157.9897461, -6449.9819336, 6070.0288086
1: -355.0456543, 252.6626129, -221.5010681, 156.6644440, -511.7100830, 474.1636963
2: -241.1246185, 409.6411743, -151.0727539, 256.3129272, -497.4375305, 560.7139282
3: -297.7525330, 606.7221069, -186.0557861, 376.1578064, -673.9102783, 792.7778931
4: -237.0228271, 414.9800720, -146.7884064, 260.3499146, -497.3726501, 561.7684937

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326227, upper bound: 4711.4326604
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326755
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4223.2407227, 3403.4516602, -4238.5522461, 3413.1520996, -7617.9809570, 7623.4780273
1: -351.0947876, 249.1456909, -352.1457825, 250.0003510, -599.6099243, 599.8065796
2: -238.3108063, 405.0757751, -239.1118622, 406.2689819, -643.3260498, 642.9334717
3: -294.0272827, 598.7977905, -294.9869385, 600.7194214, -893.0008545, 892.0568848
4: -233.9154510, 410.7941895, -234.7031097, 411.9863281, -643.9989624, 643.5853882

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4298.6142578, 3449.6342773, -4231.4921875, 3408.5009766, -7691.7500000, 7664.5419922
1: -356.1275635, 253.2917786, -351.6560364, 249.5961304, -604.4514160, 603.4839478
2: -241.8614502, 411.0473633, -238.7486267, 405.6988831, -646.2457886, 648.3273315
3: -298.4373779, 608.0323486, -294.5545349, 599.8159180, -896.4843140, 900.9838257
4: -237.5672913, 416.4108276, -234.3533936, 411.4193115, -647.0786743, 648.8582764

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4223.2407227, 3403.4516602, -4225.1215820, 3397.4221191, -7604.5595703, 7612.1054688
1: -351.0947876, 249.1456909, -350.5905762, 249.0515442, -598.8601074, 598.4500732
2: -238.3108063, 405.0757751, -238.1571045, 404.5286255, -641.8430786, 642.5216675
3: -294.0272827, 598.7977905, -293.9518738, 598.6807251, -891.3007812, 891.6753540
4: -233.9154510, 410.7941895, -233.9711151, 410.0380859, -642.4381104, 643.3056641

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326596, upper bound: 4711.4326344
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326597, upper bound: 4711.4326342
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4298.6142578, 3449.6342773, -4218.1420898, 3392.7187500, -7678.2929688, 7653.2675781
1: -356.1275635, 253.2917786, -350.0980835, 248.6480713, -603.7028809, 602.1386719
2: -241.8614502, 411.0473633, -237.7939606, 403.9533081, -644.7589111, 647.9161987
3: -298.4373779, 608.0323486, -293.5190430, 597.7753296, -894.7900391, 900.6052246
4: -237.5672913, 416.4108276, -233.6184692, 409.4677429, -645.5160522, 648.5809937

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326133, upper bound: 4711.4326204
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326558, upper bound: 4711.4326204
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4209.2080078, 3387.2622070, -4238.5522461, 3413.1520996, -7606.0185547, 7609.6162109
1: -349.4930725, 248.1587067, -352.1457825, 250.0003510, -598.2184448, 599.0198364
2: -237.3243256, 403.2819519, -239.1118622, 406.2689819, -642.8809814, 641.3979492
3: -292.9544067, 596.6747437, -294.9869385, 600.7194214, -892.5839233, 890.2779541
4: -233.1558380, 408.7913513, -234.7031097, 411.9863281, -643.6897583, 641.9718018

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326600
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326600
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4291.9921875, 3437.1938477, -4231.4921875, 3408.5009766, -7686.9873047, 7655.4497070
1: -355.0456543, 252.6626129, -351.6560364, 249.5961304, -603.5698853, 603.0969849
2: -241.1246185, 409.6411743, -238.7486267, 405.6988831, -646.0558472, 647.2983398
3: -297.7525330, 606.7221069, -294.5545349, 599.8159180, -896.4609985, 900.0093384
4: -237.0228271, 414.9800720, -234.3533936, 411.4193115, -647.0488892, 647.8123779

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326342, upper bound: 4711.4326601
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326342, upper bound: 4711.4326603
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4209.2080078, 3387.2622070, -4225.1215820, 3397.4221191, -7592.9848633, 7598.6303711
1: -349.4930725, 248.1587067, -350.5905762, 249.0515442, -597.5050049, 597.6996460
2: -237.3243256, 403.2819519, -238.1571045, 404.5286255, -641.4655762, 641.0538330
3: -292.9544067, 596.6747437, -293.9518738, 598.6807251, -890.9764404, 889.9890747
4: -233.1558380, 408.7913513, -233.9711151, 410.0380859, -642.2058105, 641.7689209

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4291.9921875, 3437.1938477, -4218.1420898, 3392.7187500, -7673.9238281, 7644.5722656
1: -355.0456543, 252.6626129, -350.0980835, 248.6480713, -602.8582764, 601.7883301
2: -241.1246185, 409.6411743, -237.7939606, 403.9533081, -644.6356812, 646.9553833
3: -297.7525330, 606.7221069, -293.5190430, 597.7753296, -894.8587646, 899.7238159
4: -237.0228271, 414.9800720, -233.6184692, 409.4677429, -645.5631714, 647.6124268

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326134, upper bound: 4711.4326543
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326741, upper bound: 4711.4326741
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.89 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326302, upper bound: 4711.4326434
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326302, upper bound: 4711.4326435
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326303, upper bound: 4711.4326433
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326303, upper bound: 4711.4326434
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326679
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326681
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326946
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326946
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326508, upper bound: 4711.4326300
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326434, upper bound: 4711.4326302
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326681, upper bound: 4711.4326796
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326993
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326961
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326961
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326973
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326961, upper bound: 4711.4326995
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326713, upper bound: 4711.4326836
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326836
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326815
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326714, upper bound: 4711.4326814
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326932, upper bound: 4711.4326572
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326930, upper bound: 4711.4326571
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326962, upper bound: 4711.4326952
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326963, upper bound: 4711.4326952
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326632, upper bound: 4711.4326231
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326342, upper bound: 4711.4326232
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326632, upper bound: 4711.4326494
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326499
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326631, upper bound: 4711.4326228
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326231
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326631, upper bound: 4711.4326681
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326681
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326086, upper bound: 4711.4326584
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326116, upper bound: 4711.4326525
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326086, upper bound: 4711.4326204
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326116, upper bound: 4711.4326204
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326114, upper bound: 4711.4326590
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326453, upper bound: 4711.4326607
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326118, upper bound: 4711.4326204
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326459, upper bound: 4711.4326205
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326828, upper bound: 4711.4326707
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326808, upper bound: 4711.4326707
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326570, upper bound: 4711.4326978
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4327016
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326680
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326227, upper bound: 4711.4326849
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326227, upper bound: 4711.4326604
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326232, upper bound: 4711.4326755
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326344
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326596, upper bound: 4711.4326344
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326597, upper bound: 4711.4326342
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326133, upper bound: 4711.4326204
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326558, upper bound: 4711.4326204
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326600
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326600
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326342, upper bound: 4711.4326601
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326342, upper bound: 4711.4326603
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326755, upper bound: 4711.4326755
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326134, upper bound: 4711.4326543
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -4711.4326741, upper bound: 4711.4326741

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2454.7104492, 2012.9052734, -2570.8798828, 2107.0661621, -4561.7763672, 4583.7851562
1: -206.4333344, 146.2502899, -216.1114807, 153.1365662, -359.5698242, 362.3617554
2: -141.0152283, 239.7950745, -147.4459991, 250.4987640, -391.5139465, 387.2410889
3: -173.6112213, 351.8038635, -181.3822479, 367.2756958, -540.8868408, 533.1860962
4: -137.1381531, 242.8017426, -143.1836548, 254.2208557, -391.3590088, 385.9854126

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2454.7104492, 2012.9052734, -2621.7290039, 2149.1535645, -4603.8637695, 4634.6342773
1: -206.4333344, 146.2502899, -220.5045929, 156.1694183, -362.6027527, 366.7548828
2: -141.0152283, 239.7950745, -150.8753052, 255.7206726, -396.7358704, 390.6703796
3: -173.6112213, 351.8038635, -185.5129547, 374.7010193, -548.3122559, 537.3168335
4: -137.1381531, 242.8017426, -146.5645599, 259.3107910, -396.4489441, 389.3663025

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2489.4003906, 2041.7442627, -2570.8798828, 2107.0661621, -4596.4663086, 4612.6240234
1: -209.4646606, 148.3458862, -216.1114807, 153.1365662, -362.6012268, 364.4573669
2: -143.4649353, 243.4570923, -147.4459991, 250.4987640, -393.9636536, 390.9030762
3: -176.5990448, 357.0031433, -181.3822479, 367.2756958, -543.8746948, 538.3853760
4: -139.6437836, 246.2940979, -143.1836548, 254.2208557, -393.8645935, 389.4777222

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2489.4003906, 2041.7442627, -2621.7290039, 2149.1535645, -4638.5532227, 4663.4731445
1: -209.4646606, 148.3458862, -220.5045929, 156.1694183, -365.6340942, 368.8504639
2: -143.4649353, 243.4570923, -150.8753052, 255.7206726, -399.1856079, 394.3323975
3: -176.5990448, 357.0031433, -185.5129547, 374.7010193, -551.3000488, 542.5161133
4: -139.6437836, 246.2940979, -146.5645599, 259.3107910, -398.9545898, 392.8586121

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2371.5834961, 1956.7266846, -2412.5383301, 1979.5397949, -4351.1230469, 4369.2651367
1: -200.5023193, 141.4376984, -202.9758911, 143.7454224, -344.2477112, 344.4135742
2: -136.6167145, 232.6844788, -138.6174927, 235.7963562, -372.4130859, 371.3019714
3: -168.3379211, 340.8910217, -170.6809387, 345.9007263, -514.2386475, 511.5719604
4: -132.8790741, 235.8761139, -134.7866974, 238.7606049, -371.6396484, 370.6627502

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326654, upper bound: 4711.4326589
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326709, upper bound: 4711.4326624
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2371.5834961, 1956.7266846, -2573.3046875, 2111.3178711, -4482.9013672, 4530.0312500
1: -200.5023193, 141.4376984, -216.6677551, 153.1012268, -353.6035461, 358.1054382
2: -136.6167145, 232.6844788, -147.6276093, 250.7150116, -387.3317261, 380.3120728
3: -168.3379211, 340.8910217, -181.8843842, 367.8164062, -536.1542969, 522.7753906
4: -132.8790741, 235.8761139, -143.5056458, 254.6773224, -387.5563965, 379.3817749

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326654, upper bound: 4711.4326587
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326710, upper bound: 4711.4326623
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2497.7451172, 2065.9453125, -2405.7148438, 1973.9224854, -4471.6669922, 4471.6601562
1: -211.6912079, 148.9949188, -202.4128418, 143.3861847, -355.0773926, 351.4077759
2: -144.2149658, 245.2859650, -138.3010559, 235.2109222, -379.4258728, 383.5869446
3: -177.5965576, 358.9290771, -170.2271118, 345.1089172, -522.7054443, 529.1560669
4: -140.0048370, 249.1302643, -134.4814148, 238.0916595, -378.0964966, 383.6116028

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326915, upper bound: 4711.4326875
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326929, upper bound: 4711.4326928
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2497.7451172, 2065.9453125, -2555.4509277, 2099.8056641, -4597.5502930, 4621.3950195
1: -211.6912079, 148.9949188, -215.3991852, 152.1680908, -363.8593140, 364.3941040
2: -144.2149658, 245.2859650, -146.8141937, 249.4365692, -393.6515198, 392.1000366
3: -177.5965576, 358.9290771, -180.8492584, 365.7245483, -543.3211060, 539.7781982
4: -140.0048370, 249.1302643, -142.6375732, 253.2702026, -393.2750244, 391.7677612

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326915, upper bound: 4711.4326875
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326929, upper bound: 4711.4326929
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -2454.7104492, 2012.9052734, -4621.0854492, 4592.3027344
1: -219.3154755, 155.3539429, -206.4333344, 146.2502899, -365.5657654, 361.7872620
2: -149.6423492, 253.9204102, -141.0152283, 239.7950745, -389.4374390, 394.9355774
3: -184.0760956, 372.3533936, -173.6112213, 351.8038635, -535.8799438, 545.9645386
4: -145.1245575, 257.9199524, -137.1381531, 242.8017426, -387.9263000, 395.0581055

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2585.5334473, 2122.4616699, -2489.4003906, 2041.7442627, -4627.2768555, 4611.8608398
1: -217.7060852, 154.0397186, -209.4646606, 148.3458862, -366.0519714, 363.5043945
2: -148.4524384, 252.0032959, -143.4649353, 243.4570923, -391.9095459, 395.4682312
3: -182.6445312, 369.4149170, -176.5990448, 357.0031433, -539.6477051, 546.0139160
4: -143.9799194, 256.0478821, -139.6437836, 246.2940979, -390.2739563, 395.6915894

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326434, upper bound: 4711.4326302
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326434, upper bound: 4711.4326302
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2573.3046875, 2111.3178711, -2371.5834961, 1956.7266846, -4530.0307617, 4482.9013672
1: -216.6677551, 153.1012268, -200.5023193, 141.4376984, -358.1054382, 353.6035461
2: -147.6276093, 250.7150116, -136.6167145, 232.6844788, -380.3120728, 387.3317261
3: -181.8843842, 367.8164062, -168.3379211, 340.8910217, -522.7753906, 536.1542969
4: -143.5056458, 254.6773224, -132.8790741, 235.8761139, -379.3817749, 387.5563965

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326661, upper bound: 4711.4326496
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326681, upper bound: 4711.4326796
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2555.4509277, 2099.8056641, -2497.7451172, 2065.9453125, -4621.3950195, 4597.5502930
1: -215.3991852, 152.1680908, -211.6912079, 148.9949188, -364.3941040, 363.8593140
2: -146.8141937, 249.4365692, -144.2149658, 245.2859650, -392.1000366, 393.6514893
3: -180.8492584, 365.7245483, -177.5965576, 358.9290771, -539.7781982, 543.3211060
4: -142.6375732, 253.2702026, -140.0048370, 249.1302643, -391.7677612, 393.2750244

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326972
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326946, upper bound: 4711.4326989
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -2608.1801758, 2137.5925293, -4745.7724609, 4745.7724609
1: -219.3154755, 155.3539429, -219.3154755, 155.3539429, -374.6693726, 374.6693726
2: -149.6423492, 253.9204102, -149.6423492, 253.9204102, -403.5627441, 403.5627441
3: -184.0760956, 372.3533936, -184.0760956, 372.3533936, -556.4293823, 556.4294434
4: -145.1245575, 257.9199524, -145.1245575, 257.9199524, -403.0444946, 403.0444946

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326684, upper bound: 4711.4326191
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326957, upper bound: 4711.4326957
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2608.1801758, 2137.5925293, -2604.8234863, 2132.1284180, -4740.3085938, 4742.4160156
1: -219.3154755, 155.3539429, -218.8876343, 154.9284515, -374.2438660, 374.2415466
2: -149.6423492, 253.9204102, -149.2814026, 253.3880920, -403.0304565, 403.2018127
3: -184.0760956, 372.3533936, -183.8647308, 371.9350586, -556.0111694, 556.2180786
4: -145.1245575, 257.9199524, -145.1332245, 257.2478333, -402.3723755, 403.0531616

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326941, upper bound: 4711.4326931
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326941, upper bound: 4711.4326941
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2604.8234863, 2132.1284180, -2608.1801758, 2137.5925293, -4742.4160156, 4740.3085938
1: -218.8876343, 154.9284515, -219.3154755, 155.3539429, -374.2415466, 374.2438660
2: -149.2814026, 253.3880920, -149.6423492, 253.9204102, -403.2018127, 403.0304565
3: -183.8647308, 371.9350586, -184.0760956, 372.3533936, -556.2181396, 556.0111694
4: -145.1332245, 257.2478333, -145.1245575, 257.9199524, -403.0531616, 402.3723450

Time for backsubstitution: 3.06 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.03 + 415.02 = 420.06 seconds
