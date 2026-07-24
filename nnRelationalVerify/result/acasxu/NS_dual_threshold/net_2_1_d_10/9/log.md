## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 146.59001129824


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348)
1: (-57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504)
2: (-47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090)
3: (-74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187)
4: (-62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 2.11 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -146.7073772, upper bound: 146.7073772

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7072596, upper bound: 146.7071560
time: 0.64 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7072596, upper bound: 146.7073184
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.41 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -146.7072596, upper bound: 146.7071560
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -146.7072596, upper bound: 146.7073184

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -68.5975342, 90.5742798, -64.3319321, 84.6784592, -153.2760010, 154.9062042
1: -54.5450630, 74.8445969, -51.0370865, 69.9069901, -124.4520569, 125.8816681
2: -45.0074158, 75.3503265, -42.1236229, 70.3498459, -115.3572617, 117.4739532
3: -71.1039505, 90.5065079, -66.5427322, 84.5768204, -155.6807709, 157.0492096
4: -59.6542358, 100.7067413, -55.8578415, 93.9662170, -153.6204529, 156.5645447

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7071295
time: 0.68 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7071560
time: 0.89 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -69.9882812, 92.4160843, -99.9843826, 135.2810211, -205.2693024, 192.4004669
1: -55.6413651, 76.3144836, -81.3746414, 113.7267685, -169.3681335, 157.6891022
2: -45.8950882, 76.8835754, -66.7984772, 113.2908783, -159.1859741, 143.6820526
3: -72.5003738, 92.2665482, -106.5640945, 137.4487457, -209.9491119, 198.8306427
4: -60.7959023, 102.7722015, -88.4699783, 151.9620972, -212.7579956, 191.2421875

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7071560, upper bound: 146.7072596
time: 0.68 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7071560, upper bound: 146.7073184
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.15 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7071295
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7071560
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -146.7071560, upper bound: 146.7072596
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -146.7071560, upper bound: 146.7073184

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -64.3319321, 84.6784592, -64.3319321, 84.6784592, -149.0103912, 149.0103912
1: -51.0370865, 69.9069901, -51.0370865, 69.9069901, -120.9440536, 120.9440536
2: -42.1236229, 70.3498459, -42.1236229, 70.3498459, -112.4734650, 112.4734650
3: -66.5427322, 84.5768204, -66.5427322, 84.5768204, -151.1195374, 151.1195526
4: -55.8578415, 93.9662170, -55.8578415, 93.9662170, -149.8240662, 149.8240662

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049443, upper bound: 146.7054185
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049349, upper bound: 146.7049349
time: 0.68 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -99.4544067, 134.6990662, -64.3319321, 84.6784592, -184.1328735, 199.0309906
1: -81.0188828, 113.2551346, -51.0370865, 69.9069901, -150.9258270, 164.2922058
2: -66.5035629, 112.7858276, -42.1236229, 70.3498459, -136.8533936, 154.9094543
3: -106.1131744, 136.8876648, -66.5427322, 84.5768204, -190.6899872, 203.4303741
4: -88.0779190, 151.3189240, -55.8578415, 93.9662170, -182.0441284, 207.1767578

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7069938, upper bound: 146.7070465
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7071560
time: 0.63 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -64.3319321, 84.6784592, -99.9843826, 135.2810211, -199.6129456, 184.6628418
1: -51.0370865, 69.9069901, -81.3746414, 113.7267685, -164.7638550, 151.2816162
2: -42.1236229, 70.3498459, -66.7984772, 113.2908783, -155.4145050, 137.1483154
3: -66.5427322, 84.5768204, -106.5640945, 137.4487457, -203.9914856, 191.1409149
4: -55.8578415, 93.9662170, -88.4699783, 151.9620972, -207.8199310, 182.4361877

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7070465, upper bound: 146.7070956
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7072420
time: 0.74 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -99.9843826, 135.2810211, -99.9843826, 135.2810211, -235.2654114, 235.2654114
1: -81.3746414, 113.7267685, -81.3746414, 113.7267685, -195.1014099, 195.1014099
2: -66.7984772, 113.2908783, -66.7984772, 113.2908783, -180.0893555, 180.0893555
3: -106.5640945, 137.4487457, -106.5640945, 137.4487457, -244.0128479, 244.0128479
4: -88.4699783, 151.9620972, -88.4699783, 151.9620972, -240.4320679, 240.4320679

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6991399, upper bound: 146.7068934
time: 1.15 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6988655, upper bound: 146.6988655
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.59 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.7049443, upper bound: 146.7054185
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.7049349, upper bound: 146.7049349
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.7069938, upper bound: 146.7070465
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7071560
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.7070465, upper bound: 146.7070956
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.7071295, upper bound: 146.7072420
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.6991399, upper bound: 146.7068934
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -146.6988655, upper bound: 146.6988655

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -49.0153198, 64.4453812, -59.7289200, 78.6254425, -127.6407547, 124.1743011
1: -38.7037392, 52.9584770, -47.3727722, 64.8702087, -103.5739441, 100.3312531
2: -31.8804531, 53.4162140, -39.0631599, 65.3044434, -97.1848984, 92.4793701
3: -50.7389641, 64.0345078, -61.8075066, 78.4936905, -129.2326355, 125.8420029
4: -42.2791710, 71.1144638, -51.8189278, 87.1818237, -129.4609985, 122.9333954

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6509824, upper bound: 146.6907537
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7044342, upper bound: 146.7048772
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -56.3686295, 73.2173462, -63.5859642, 83.6049423, -139.9735718, 136.8033142
1: -44.3196907, 60.3661880, -50.4055061, 69.0131683, -113.3328552, 110.7716980
2: -36.6174927, 60.7190895, -41.6073303, 69.4370728, -106.0545654, 102.3264160
3: -57.4893456, 73.0946579, -65.6973114, 83.5005722, -140.9898834, 138.7919312
4: -48.4997177, 80.8013077, -55.1673241, 92.7279510, -141.2276611, 135.9686279

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6482515, upper bound: 146.6868895
time: 0.70 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7039672, upper bound: 146.7039672
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -96.4205551, 130.5262299, -72.6220169, 95.7993546, -192.2198944, 203.1482086
1: -78.5590591, 109.7871170, -58.1416702, 79.4542084, -158.0132294, 167.9287872
2: -64.4667130, 109.2040024, -47.9720421, 79.3330917, -143.7997894, 157.1760406
3: -102.9183350, 132.7065887, -75.5190659, 96.3835678, -199.3019104, 208.2256317
4: -85.3607559, 146.5253754, -63.4548912, 106.0752792, -191.4360352, 209.9802704

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
time: 0.69 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -98.3923721, 133.2187653, -61.5820961, 80.8930740, -179.2854462, 194.8008575
1: -80.1524963, 112.0325623, -48.8429871, 66.7791367, -146.9315948, 160.8755493
2: -65.7851715, 111.5157166, -40.3099174, 67.0830841, -132.8682404, 151.8256226
3: -104.9796753, 135.4198151, -63.6634254, 80.8456039, -185.8252411, 199.0832367
4: -87.1117783, 149.6127625, -53.4593544, 89.6017685, -176.7135468, 203.0720978

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6091958
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -72.6220169, 95.7993546, -96.9192276, 131.0746460, -203.6966553, 192.7185211
1: -58.1416702, 79.4542084, -78.8941803, 110.2308807, -168.3725586, 158.3483582
2: -47.9720421, 79.3330917, -64.7440948, 109.6803970, -157.6524353, 144.0771790
3: -75.5190659, 96.3835678, -103.3427582, 133.2322693, -208.7513123, 199.7263184
4: -63.4548912, 106.0752792, -85.7315674, 147.1318207, -210.5867157, 191.8068542

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
time: 0.99 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045606, upper bound: 146.6982580
time: 0.94 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -61.5820961, 80.8930740, -98.9104691, 133.7886353, -195.3707123, 179.8035431
1: -48.8429871, 66.7791367, -80.5008850, 112.4938049, -161.3367615, 147.2799988
2: -40.3099174, 67.0830841, -66.0737152, 112.0105591, -152.3204651, 133.1567688
3: -63.6634254, 80.8456039, -105.4211044, 135.9684601, -199.6318817, 186.2666779
4: -53.4593544, 89.6017685, -87.4973907, 150.2430573, -203.7023926, 177.0991516

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6442341, upper bound: 146.5800372
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6091958, upper bound: 146.5791316
time: 0.96 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -87.2272949, 117.6857300, -96.0946426, 129.8014526, -217.0287018, 213.7803040
1: -70.7591782, 99.1173935, -78.1114273, 109.1527100, -179.9118958, 177.2288208
2: -58.1809082, 98.5051727, -64.1406021, 108.6790009, -166.8598938, 162.6457825
3: -92.7398605, 119.7210159, -102.2866898, 131.8991699, -224.6390381, 222.0077057
4: -77.0188217, 132.0534973, -84.9191284, 145.7419128, -222.7607422, 216.9726257

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7023709
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6962331, upper bound: 146.7047484
time: 1.10 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -92.2137146, 124.4555054, -98.7997818, 133.5820770, -225.7957458, 223.2552490
1: -74.9719315, 104.3598251, -80.3796310, 112.2660370, -187.2379761, 184.7394104
2: -61.4643211, 104.1060867, -65.9770889, 111.8481445, -173.3124695, 170.0831757
3: -98.1501007, 126.2656326, -105.2461700, 135.7031860, -233.8532867, 231.5117950
4: -81.3528442, 139.4653168, -87.3691177, 149.9953766, -231.3482208, 226.8344421

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
time: 1.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.83 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6509824, upper bound: 146.6907537
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.7044342, upper bound: 146.7048772
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6482515, upper bound: 146.6868895
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.7039672, upper bound: 146.7039672
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6091958
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.7045606, upper bound: 146.6982580
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6442341, upper bound: 146.5800372
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6091958, upper bound: 146.5791316
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7023709
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6962331, upper bound: 146.7047484
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -46.3293381, 60.8002968, -56.3767014, 73.9221878, -120.2515259, 117.1769791
1: -36.3551140, 49.9204254, -44.0700417, 60.9111328, -97.2662506, 93.9904633
2: -29.9544258, 50.4699440, -36.4160385, 61.4465027, -91.4009247, 86.8859634
3: -47.7256432, 60.4482040, -57.5296898, 73.6398239, -121.3654633, 117.9778900
4: -39.7921448, 67.0745087, -48.4613571, 81.7342148, -121.5263596, 115.5358658

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077893, upper bound: 146.6650062
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6436317, upper bound: 146.6867315
time: 0.97 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -49.0153198, 64.4453812, -54.5269852, 71.6537857, -120.6690979, 118.9723358
1: -38.7037392, 52.9584770, -43.0232201, 58.9105873, -97.6143265, 95.9816971
2: -31.8804531, 53.4162140, -35.4819298, 59.6139679, -91.4944229, 88.8981400
3: -50.7389641, 64.0345078, -56.1561661, 71.2113724, -121.9503326, 120.1906738
4: -42.2791710, 71.1144638, -47.1082344, 79.4731979, -121.7523651, 118.2227020

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7033884, upper bound: 146.7035217
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7034953, upper bound: 146.7033496
time: 1.06 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -54.8958054, 71.1820526, -78.0862732, 103.4037628, -158.2995148, 149.2683258
1: -43.0605087, 58.6761551, -61.6592751, 85.4745255, -128.5350342, 120.3354340
2: -35.5957642, 59.0308304, -51.0817909, 86.6981354, -122.2938995, 110.1126251
3: -55.8766708, 71.0376053, -80.5626373, 102.9353790, -158.8120270, 151.6002350
4: -47.1604385, 78.5140762, -67.8289261, 115.9548492, -163.1152802, 146.3429871

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6079052, upper bound: 146.6662134
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6449426, upper bound: 146.6853101
time: 0.98 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -55.1748238, 71.5559998, -55.9457092, 73.1362991, -128.3110962, 127.5017090
1: -43.2820854, 58.9334335, -43.8901100, 59.8942528, -103.1763229, 102.8235321
2: -35.7842560, 59.3611984, -36.2693100, 60.7149773, -96.4992294, 95.6304932
3: -56.1523743, 71.3485489, -57.1808243, 72.4404526, -128.5928192, 128.5293579
4: -47.3917542, 78.9590225, -48.0748367, 80.9023895, -128.2941284, 127.0338440

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6868895, upper bound: 146.6482515
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6868895, upper bound: 146.7039672
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -80.8092880, 109.5798798, -67.3503113, 88.6653442, -169.4745789, 176.9301758
1: -65.7901306, 92.2695694, -53.7942467, 73.4245453, -139.2146759, 146.0638123
2: -54.0285568, 91.7289581, -44.3938637, 73.3622360, -127.3907852, 136.1228027
3: -86.4879227, 111.4742813, -69.8194656, 89.0618744, -175.5497894, 181.2937469
4: -71.4843521, 123.1768723, -58.7068520, 98.0115585, -169.4959106, 181.8837280

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6865168, upper bound: 146.7031794
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7049301
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7049301
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -85.9650116, 115.0960159, -71.8572540, 94.7089691, -180.6739349, 186.9532776
1: -69.5866852, 96.8072739, -57.4993629, 78.5447540, -148.1314392, 154.3066254
2: -57.2010880, 96.0271301, -47.4487267, 78.4072342, -135.6083221, 143.4758606
3: -90.8757477, 117.0995865, -74.6647644, 95.2840958, -186.1598511, 191.7643433
4: -75.5781937, 128.6557465, -62.7583885, 104.8261566, -180.4043427, 191.4141388

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6866469, upper bound: 146.7035489
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
time: 1.15 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -96.3778687, 130.4256287, -61.5820961, 80.8930740, -177.2709351, 192.0077209
1: -78.4845200, 109.7177048, -48.8429871, 66.7791367, -145.2636566, 158.5606689
2: -64.4200897, 109.1588745, -40.3099174, 67.0830841, -131.5031281, 149.4687805
3: -102.8104630, 132.6143799, -63.6634254, 80.8456039, -183.6560516, 196.2778015
4: -85.2936935, 146.4570923, -53.4593544, 89.6017685, -174.8954620, 199.9164429

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5730509, upper bound: 146.6337493
time: 0.68 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -93.5869446, 128.4192505, -58.9537010, 77.5160522, -171.1029968, 187.3729401
1: -76.6929016, 107.8488693, -46.7003593, 63.9323540, -140.6252594, 154.5492096
2: -62.7895012, 108.0062561, -38.5369148, 64.3624420, -127.1519394, 146.5431671
3: -100.7713318, 130.1181335, -60.9463577, 77.3625793, -178.1339111, 191.0644684
4: -83.3629837, 144.9443054, -51.1340370, 85.9329376, -169.2959290, 196.0783386

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5730509, upper bound: 146.5933762
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6090476
time: 0.88 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -67.3503113, 88.6653442, -81.1741180, 109.9714966, -177.3217621, 169.8394623
1: -53.7942467, 73.4245453, -66.0344543, 92.5876770, -146.3819275, 139.4589691
2: -44.3938637, 73.3622360, -54.2302322, 92.0692978, -136.4631653, 127.5924683
3: -69.8194656, 89.0618744, -86.7947998, 111.8501129, -181.6695709, 175.8566589
4: -58.7068520, 98.0115585, -71.7543182, 123.6076584, -182.3144836, 169.7658691

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031794, upper bound: 146.6865168
time: 0.61 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049301, upper bound: 146.6981470
time: 0.66 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049301, upper bound: 146.6981470
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -71.8572540, 94.7089691, -86.3517303, 115.5211716, -187.3784180, 181.0606995
1: -57.4993629, 78.5447540, -69.8458557, 97.1507568, -154.6501160, 148.3906097
2: -47.4487267, 78.4072342, -57.4162140, 96.3974762, -143.8461914, 135.8234558
3: -74.6647644, 95.2840958, -91.2034454, 117.5054779, -192.1701965, 186.4875336
4: -62.7583885, 104.8261566, -75.8674469, 129.1265411, -191.8849335, 180.6935883

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7035489, upper bound: 146.6866469
time: 0.64 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6923657, upper bound: 146.6860786
time: 0.63 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -61.5820961, 80.8930740, -96.8776779, 130.9754333, -192.5575104, 177.7707520
1: -48.8429871, 66.7791367, -78.8214645, 110.1622162, -159.0051422, 145.6006012
2: -40.3099174, 67.0830841, -64.6989212, 109.6365280, -149.9464264, 131.7819824
3: -63.6634254, 80.8456039, -103.2365952, 133.1394043, -196.8028259, 184.0821533
4: -53.4593544, 89.6017685, -85.6650848, 147.0656891, -200.5250092, 175.2668304

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6337493, upper bound: 146.5730509
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6442964, upper bound: 146.5801684
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -58.9537010, 77.5160522, -93.8404312, 128.6947021, -187.6484070, 171.3564758
1: -46.7003593, 63.9323540, -76.8610382, 108.0722046, -154.7725372, 140.7933960
2: -38.5369148, 64.3624420, -62.9293518, 108.2447891, -146.7817078, 127.2917938
3: -60.9463577, 77.3625793, -100.9839172, 130.3835449, -191.3298798, 178.3464813
4: -51.1340370, 85.9329376, -83.5501862, 145.2469025, -196.3809357, 169.4831085

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5734511
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6090476, upper bound: 146.5791316
time: 1.05 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -84.2579193, 113.6107483, -80.0436249, 108.2236176, -192.4815063, 193.6543732
1: -68.3424759, 95.7370911, -64.9801178, 91.1160583, -159.4584961, 160.7171631
2: -56.2112274, 95.0474777, -53.4079285, 90.6169052, -146.8281097, 148.4554138
3: -89.5892487, 115.6441956, -85.3921585, 110.0331650, -199.6224060, 201.0363312
4: -74.3736038, 127.4047089, -70.6586304, 121.6472244, -196.0208282, 198.0633392

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6799665, upper bound: 146.7023709
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6799665, upper bound: 146.7023709
time: 0.91 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -86.2923279, 116.3344193, -84.6216431, 112.9491501, -199.2414703, 200.9560394
1: -69.9667282, 97.9800949, -68.3226929, 94.9506531, -164.9173737, 166.3027954
2: -57.5360298, 97.3566742, -56.1938438, 94.2676239, -151.8036499, 153.5505066
3: -91.6833954, 118.3517303, -89.1599655, 114.8290939, -206.5124817, 207.5116882
4: -76.1461029, 130.4945984, -74.2093201, 126.2103500, -202.3564453, 204.7039185

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6957464, upper bound: 146.7046115
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6957464, upper bound: 146.7047484
time: 1.01 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -92.2137146, 124.4555054, -96.7865067, 130.8067474, -223.0204163, 221.2419891
1: -74.9719315, 104.3598251, -78.7232590, 109.9546509, -184.9265747, 183.0830383
2: -61.4643211, 104.1060867, -64.6193619, 109.5034485, -170.9677582, 168.7254486
3: -98.1501007, 126.2656326, -103.0897522, 132.8958893, -231.0459900, 229.3553619
4: -81.3528442, 139.4653168, -85.5665131, 146.8572845, -228.2101288, 225.0318298

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6764617, upper bound: 146.6191114
time: 0.95 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6564928, upper bound: 146.5973341
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -89.2532806, 120.7914124, -93.8257675, 128.6371765, -217.8904572, 214.6171417
1: -72.6226273, 101.2624664, -76.8269119, 107.9650726, -180.5877075, 178.0893860
2: -59.5044594, 101.1106644, -62.9012985, 108.2000275, -167.7044830, 164.0119324
3: -95.1613846, 122.4866562, -100.9363632, 130.2722931, -225.4336853, 223.4230194
4: -78.8094864, 135.4436798, -83.5092697, 145.1650696, -223.9745483, 218.9529419

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
time: 1.03 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.60 seconds
NS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6077893, upper bound: 146.6650062
NS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6436317, upper bound: 146.6867315
NS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.7033884, upper bound: 146.7035217
NS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.7034953, upper bound: 146.7033496
NS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6079052, upper bound: 146.6662134
NS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6449426, upper bound: 146.6853101
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6868895, upper bound: 146.6482515
NS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6868895, upper bound: 146.7039672
NS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7049301
NS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7049301
NS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
NS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
NS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.5730509, upper bound: 146.6337493
NS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
NS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.5730509, upper bound: 146.5933762
NS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6090476
NS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.7049301, upper bound: 146.6981470
NS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.7049301, upper bound: 146.6981470
NS_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.7035489, upper bound: 146.6866469
NS_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6923657, upper bound: 146.6860786
NS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6337493, upper bound: 146.5730509
NS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6442964, upper bound: 146.5801684
NS_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5734511
NS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6090476, upper bound: 146.5791316
NS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6799665, upper bound: 146.7023709
NS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6799665, upper bound: 146.7023709
NS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6957464, upper bound: 146.7046115
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6957464, upper bound: 146.7047484
NS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6564928, upper bound: 146.5973341
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740

## BFS NS instance: NS_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -45.0544243, 59.0877151, -63.4164619, 83.2889862, -128.3433990, 122.5041809
1: -35.2652969, 48.5281525, -48.9800644, 68.7593842, -104.0246811, 97.5081863
2: -29.0587730, 49.0974464, -40.7057571, 70.1618729, -99.2206421, 89.8031921
3: -46.3677292, 58.7543335, -64.1606293, 82.7382889, -129.1060181, 122.9149628
4: -38.6149673, 65.2119980, -54.1929169, 93.2696152, -131.8845825, 119.4049149

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6645849
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6650062
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -45.3901520, 59.3852539, -49.0284653, 63.1552391, -108.5453949, 108.4137115
1: -35.4866028, 48.7539139, -37.4323120, 51.8843918, -87.3709869, 86.1862259
2: -29.2628784, 49.3198471, -31.0997543, 52.6751099, -81.9379883, 80.4195938
3: -46.5800285, 59.0281715, -48.8514748, 62.6909065, -109.2709351, 107.8796387
4: -38.8630714, 65.5064087, -41.3561020, 69.7939453, -108.6570053, 106.8625031

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6435124, upper bound: 146.6854790
time: 1.01 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6435124, upper bound: 146.6867315
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -48.5949402, 63.6081314, -53.0561638, 69.6073227, -118.2022629, 116.6642914
1: -37.7306786, 52.4548111, -41.7604179, 57.2547417, -94.9854202, 94.2152100
2: -31.1921425, 53.2458687, -34.4820938, 57.9721451, -89.1642914, 87.7279434
3: -49.6124077, 63.3990364, -54.5498161, 69.1933517, -118.8057556, 117.9488297
4: -41.5441780, 70.5757675, -45.7792320, 77.2407303, -118.7849045, 116.3549957

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6061779, upper bound: 146.5110633
time: 0.86 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5555999, upper bound: 146.5095945
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.5288582, 56.4985695, -53.4570122, 70.1421356, -113.6709900, 109.9555817
1: -33.7135353, 46.1598244, -42.0860596, 57.5977821, -91.3113174, 88.2458801
2: -27.8262539, 46.7750587, -34.7196770, 58.3493271, -86.1755829, 81.4947205
3: -44.0973091, 55.9171486, -54.9058685, 69.6226196, -113.7199249, 110.8230133
4: -36.9151611, 62.0547600, -46.0890427, 77.7544632, -114.6696167, 108.1437988

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6448589, upper bound: 146.6835662
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6448589, upper bound: 146.7033496
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -50.2008934, 64.8055954, -68.5043335, 90.0341568, -140.2350159, 133.3099365
1: -39.0678787, 53.3663597, -53.0773087, 74.3221359, -113.3899994, 106.4436569
2: -32.3175545, 53.7746162, -44.1364021, 75.7964706, -108.1140213, 97.9110184
3: -50.7987137, 64.6656265, -69.4603424, 89.3869934, -140.1856689, 134.1259766
4: -42.8605537, 71.3362961, -58.7091751, 100.9141769, -143.7747345, 130.0454712

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5327808, upper bound: 146.6225380
time: 0.95 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5309511, upper bound: 146.5562047
time: 0.73 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -54.8958054, 71.1820526, -73.9281998, 97.8744202, -152.7701721, 145.1102600
1: -43.0605087, 58.6761551, -58.1919441, 80.5927505, -123.6532593, 116.8680954
2: -35.5957642, 59.0308304, -48.1970291, 82.0537949, -117.6495590, 107.2278519
3: -55.8766708, 71.0376053, -76.1544571, 96.9829102, -152.8595581, 147.1920624
4: -47.1604385, 78.5140762, -64.0241547, 109.7183838, -156.8788147, 142.5382080

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6439509
time: 0.59 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6853101
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -70.3638153, 92.4291382, -55.9457092, 73.1362991, -143.5001221, 148.3748322
1: -55.1202660, 76.1830978, -43.8901100, 59.8942528, -115.0145187, 120.0731964
2: -45.7413101, 77.6045532, -36.2693100, 60.7149773, -106.4562836, 113.8738632
3: -71.8991776, 91.6738815, -57.1808243, 72.4404526, -144.3396301, 148.8546906
4: -60.6913795, 103.5729828, -48.0748367, 80.9023895, -141.5937653, 151.6478271

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6472987, upper bound: 146.6481984
time: 0.92 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6472987, upper bound: 146.6481984
time: 1.77 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.6898956, 62.5317764, -55.9457092, 73.1362991, -121.8261948, 118.4774704
1: -37.5633392, 51.1040039, -43.8901100, 59.8942528, -97.4575958, 94.9941101
2: -31.1202602, 51.7925301, -36.2693100, 60.7149773, -91.8352356, 88.0618439
3: -48.7319031, 61.9018021, -57.1808243, 72.4404526, -121.1723557, 119.0826263
4: -41.1979332, 68.6522675, -48.0748367, 80.9023895, -122.1003265, 116.7270966

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6412062, upper bound: 146.6955956
time: 0.70 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6412062, upper bound: 146.7039633
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -80.8092880, 109.5798798, -58.1859283, 76.2957153, -157.1049805, 167.7658081
1: -65.7901306, 92.2695694, -46.3820801, 62.9604378, -128.7505646, 138.6516418
2: -54.0285568, 91.7289581, -38.2315903, 62.8689003, -116.8974533, 129.9605408
3: -86.4879227, 111.4742813, -60.1663933, 76.4167404, -162.9046631, 171.6406708
4: -71.4843521, 123.1768723, -50.5320625, 83.7994766, -155.2838287, 173.7089386

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -80.8092880, 109.5798798, -63.2701683, 82.4207993, -163.2300415, 172.8500519
1: -65.7901306, 92.2695694, -50.2276192, 68.2725677, -134.0626984, 142.4971771
2: -54.0285568, 91.7289581, -41.5368042, 67.8949814, -121.9235306, 133.2657623
3: -86.4879227, 111.4742813, -64.9713974, 82.8786316, -169.3665466, 176.4456787
4: -71.4843521, 123.1768723, -54.8750114, 90.6103745, -162.0947266, 178.0518799

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -85.9650116, 115.0960159, -58.1859283, 76.2957153, -162.2607269, 173.2819519
1: -69.5866852, 96.8072739, -46.3820801, 62.9604378, -132.5471191, 143.1893616
2: -57.2010880, 96.0271301, -38.2315903, 62.8689003, -120.0699844, 134.2587280
3: -90.8757477, 117.0995865, -60.1663933, 76.4167404, -167.2924652, 177.2659760
4: -75.5781937, 128.6557465, -50.5320625, 83.7994766, -159.3776550, 179.1878052

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5277178, upper bound: 146.6130842
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5264679, upper bound: 146.5702614
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -85.9650116, 115.0960159, -63.2701683, 82.4207993, -168.3858032, 178.3661804
1: -69.5866852, 96.8072739, -50.2276192, 68.2725677, -137.8592529, 147.0348969
2: -57.2010880, 96.0271301, -41.5368042, 67.8949814, -125.0960617, 137.5639343
3: -90.8757477, 117.0995865, -64.9713974, 82.8786316, -173.7543793, 182.0709839
4: -75.5781937, 128.6557465, -54.8750114, 90.6103745, -166.1885223, 183.5307617

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6763510, upper bound: 146.6395453
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
time: 0.97 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -90.4219818, 121.9592972, -58.5717506, 76.5637894, -166.9857788, 180.5310364
1: -73.1199570, 102.5635071, -46.1552048, 63.1446571, -136.2645721, 148.7187195
2: -60.1868172, 102.0159378, -38.1644173, 63.4730568, -123.6598663, 140.1803589
3: -95.8767319, 123.8952942, -60.1595421, 76.4191132, -172.2958221, 184.0548401
4: -79.7158279, 136.8176270, -50.6172104, 84.7007828, -164.4165955, 187.4348450

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6334123
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6337493
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -94.8670349, 128.3668213, -61.4845428, 80.7592316, -175.6262665, 189.8513489
1: -77.2229843, 107.9733505, -48.7606201, 66.6667099, -143.8896942, 156.7339325
2: -63.3891525, 107.4281540, -40.2427330, 66.9710388, -130.3601837, 147.6708832
3: -101.1763992, 130.4962311, -63.5568771, 80.7087936, -181.8851624, 194.0530853
4: -83.9320221, 144.1323700, -53.3707008, 89.4513931, -173.3834076, 197.5030670

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
time: 1.01 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
time: 0.62 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -86.9514771, 119.0431976, -55.9778481, 73.2439346, -160.1953583, 175.0210419
1: -70.7517090, 99.9098358, -44.0442924, 60.3464203, -131.0981293, 143.9541321
2: -58.0821304, 100.1023331, -36.4124298, 60.8250885, -118.9072189, 136.5147705
3: -93.1455078, 120.4289322, -57.4824677, 72.9918442, -166.1373596, 177.9113770
4: -77.1675339, 134.3264465, -48.3217506, 81.1048431, -158.2723694, 182.6481934

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5734511, upper bound: 146.5933762
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5734511, upper bound: 146.5933762
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -92.1462021, 126.4399643, -58.8568192, 77.3822784, -169.5284729, 185.2967682
1: -75.4836807, 106.1654816, -46.6178551, 63.8197937, -139.3034668, 152.7833405
2: -61.8041611, 106.3367386, -38.4698753, 64.2511826, -126.0553360, 144.8066101
3: -99.2066345, 128.0771484, -60.8400345, 77.2255859, -176.4322205, 188.9171753
4: -82.0615311, 142.7024078, -51.0455208, 85.7827606, -167.8442993, 193.7479248

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6090476
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6090476
time: 0.65 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -58.1859283, 76.2957153, -81.1741180, 109.9714966, -168.1574097, 157.4698334
1: -46.3820801, 62.9604378, -66.0344543, 92.5876770, -138.9697571, 128.9948883
2: -38.2315903, 62.8689003, -54.2302322, 92.0692978, -130.3008881, 117.0991287
3: -60.1663933, 76.4167404, -86.7947998, 111.8501129, -172.0164795, 163.2115479
4: -50.5320625, 83.7994766, -71.7543182, 123.6076584, -174.1397247, 155.5538025

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
time: 0.92 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -63.2701683, 82.4207993, -81.1741180, 109.9714966, -173.2416687, 163.5949097
1: -50.2276192, 68.2725677, -66.0344543, 92.5876770, -142.8152924, 134.3070221
2: -41.5368042, 67.8949814, -54.2302322, 92.0692978, -133.6061096, 122.1252136
3: -64.9713974, 82.8786316, -86.7947998, 111.8501129, -176.8215027, 169.6734314
4: -54.8750114, 90.6103745, -71.7543182, 123.6076584, -178.4826508, 162.3646851

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
time: 1.26 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -67.9854736, 89.3568497, -74.7648926, 99.6353912, -167.6208649, 164.1217346
1: -54.2914734, 74.0940018, -60.2679405, 83.9649277, -138.2563782, 134.3619232
2: -44.8189354, 73.9349289, -49.6179390, 83.0990295, -127.9179611, 123.5528641
3: -70.4506760, 89.8798904, -78.7299118, 101.4767990, -171.9274750, 168.6097717
4: -59.2695885, 98.7837753, -65.5002899, 111.1828690, -170.4524536, 164.2840576

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6866469
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6865168
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -70.6798096, 93.0840759, -78.9442596, 105.4462433, -176.1260529, 172.0283356
1: -56.5366020, 77.1314697, -63.8592796, 88.3929596, -144.9295654, 140.9907532
2: -46.6428108, 77.0236740, -52.3859863, 87.8983459, -134.5411530, 129.4096222
3: -73.3837051, 93.5942612, -83.3695145, 107.0792007, -180.4628906, 176.9637299
4: -61.6865921, 102.9384613, -69.1731491, 117.5145493, -179.2011414, 172.1116028

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6425322, upper bound: 146.5731233
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5994463, upper bound: 146.5721441
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -58.5717506, 76.5637894, -90.9309692, 122.5153961, -181.0871124, 167.4947357
1: -46.1552048, 63.1446571, -73.4620056, 103.0150375, -149.1702118, 136.6066284
2: -38.1644173, 63.4730568, -60.4697571, 102.5000763, -140.6644592, 123.9428101
3: -60.1595421, 76.4191132, -96.3089676, 124.4303055, -184.5898438, 172.7280579
4: -50.6172104, 84.7007828, -80.0910110, 137.4320221, -188.0492249, 164.7917938

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6334123, upper bound: 146.5674976
time: 0.87 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5730509
time: 1.05 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -61.4845428, 80.7592316, -95.3544235, 128.9032898, -190.3878021, 176.1136322
1: -48.7606201, 66.6667099, -77.5516357, 108.4070740, -157.1676483, 144.2183533
2: -40.2427330, 66.9710388, -63.6608925, 107.8943710, -148.1371002, 130.6319275
3: -63.5568771, 80.7087936, -101.5921097, 131.0086975, -194.5655670, 182.3008423
4: -53.3707008, 89.4513931, -84.2938538, 144.7262115, -198.0968933, 173.7452393

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6442964, upper bound: 146.5801684
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6337493, upper bound: 146.5801684
time: 0.65 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -55.9778481, 73.2439346, -87.2007294, 119.3170547, -175.2949066, 160.4446106
1: -44.0442924, 60.3464203, -70.9196091, 100.1323242, -144.1766205, 131.2659912
2: -36.4124298, 60.8250885, -58.2212296, 100.3403320, -136.7527618, 119.0463104
3: -57.4824677, 72.9918442, -93.3581390, 120.6934967, -178.1759338, 166.3499756
4: -48.3217506, 81.1048431, -77.3527527, 134.6280518, -182.9497986, 158.4575958

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5734511
time: 1.22 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5730509
time: 0.63 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -58.8568192, 77.3822784, -92.3872986, 126.7017288, -185.5585327, 169.7695770
1: -46.6178551, 63.8197937, -75.6435089, 106.3778534, -152.9957123, 139.4632874
2: -38.4698753, 64.2511826, -61.9371109, 106.5636292, -145.0334930, 126.1882858
3: -60.8400345, 77.2255859, -99.4087219, 128.3294067, -189.1694336, 176.6343079
4: -51.0455208, 85.7827606, -82.2395782, 142.9899902, -194.0355072, 168.0223236

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6090476, upper bound: 146.5791316
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6090476, upper bound: 146.5791316
time: 0.68 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -84.2579193, 113.6107483, -72.3207779, 97.6563263, -181.9141541, 185.9315186
1: -68.3424759, 95.7370911, -58.4869423, 82.3626709, -150.7051392, 154.2240143
2: -56.2112274, 95.0474777, -48.1558838, 81.7584534, -137.9696503, 143.2033539
3: -89.5892487, 115.6441956, -76.9195251, 99.3732681, -188.9625244, 192.5637054
4: -74.3736038, 127.4047089, -63.7550850, 109.6183014, -183.9918976, 191.1597900

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6799665, upper bound: 146.7023709
time: 1.32 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7020957
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798467, upper bound: 146.7023709
time: 0.68 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -84.2579193, 113.6107483, -75.0877838, 101.5556335, -185.8135376, 188.6985168
1: -68.3424759, 95.7370911, -61.0108986, 85.2025681, -153.5450134, 156.7479858
2: -56.2112274, 95.0474777, -50.0403366, 84.9559402, -141.1671753, 145.0877991
3: -89.5892487, 115.6441956, -80.2089539, 103.0284119, -192.6176453, 195.8531494
4: -74.3736038, 127.4047089, -66.1934967, 113.9446564, -188.3182526, 193.5982056

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6799665, upper bound: 146.7023709
time: 0.88 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7020957
time: 0.62 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7023709
time: 0.61 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -72.3207779, 97.6563263, -84.6216431, 112.9491501, -185.2699280, 182.2778931
1: -58.4869423, 82.3626709, -68.3226929, 94.9506531, -153.4375763, 150.6853638
2: -48.1558838, 81.7584534, -56.1938438, 94.2676239, -142.4235077, 137.9522858
3: -76.9195251, 99.3732681, -89.1599655, 114.8290939, -191.7486115, 188.5332336
4: -63.7550850, 109.6183014, -74.2093201, 126.2103500, -189.9654388, 183.8276062

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7046115
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7046115
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -76.6164932, 102.1269913, -84.6216431, 112.9491501, -189.5656433, 186.7485809
1: -61.7346001, 85.9981766, -68.3226929, 94.9506531, -156.6852570, 154.3208618
2: -50.8304214, 85.2304993, -56.1938438, 94.2676239, -145.0980530, 141.4243469
3: -80.6253738, 103.9331131, -89.1599655, 114.8290939, -195.4544525, 193.0930786
4: -67.1096649, 114.0315094, -74.2093201, 126.2103500, -193.3200073, 188.2407990

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798467, upper bound: 146.7047320
time: 0.99 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6798467, upper bound: 146.7047320
time: 0.69 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -87.9044342, 118.3958511, -93.7582779, 126.6526413, -214.5570679, 212.1541290
1: -71.4883118, 99.2353973, -76.2749176, 106.5015335, -177.9898376, 175.5103149
2: -58.6920929, 98.8520355, -62.5941658, 105.9324646, -164.6245575, 161.4461975
3: -93.5039215, 120.1097260, -99.9107895, 128.7380829, -222.2420044, 220.0205078
4: -77.6409378, 132.5579834, -82.8577957, 142.0824280, -219.7233582, 215.4157562

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6564929, upper bound: 146.5973341
time: 0.74 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6564929, upper bound: 146.5973341
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -88.2007523, 118.9087067, -95.7246552, 129.3341522, -217.5348816, 214.6333313
1: -71.7228775, 99.7583771, -77.8616943, 108.7375565, -180.4604187, 177.6200104
2: -58.7754822, 99.3453369, -63.9038506, 108.2392349, -167.0147095, 163.2491913
3: -93.9081726, 120.7625427, -101.9636459, 131.4375763, -225.3457184, 222.7261963
4: -77.7578506, 133.0831146, -84.5994949, 145.1608429, -222.9187012, 217.6826172

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -90.2200241, 121.6943283, -93.8257675, 128.6371765, -218.8572083, 215.5200958
1: -73.3221207, 102.0556412, -76.8269119, 107.9650726, -181.2871399, 178.8825531
2: -60.1172523, 101.7760468, -62.9012985, 108.2000275, -168.3172760, 164.6773376
3: -95.9929504, 123.4554672, -100.9363632, 130.2722931, -226.2652435, 224.3918152
4: -79.5604401, 136.3418121, -83.5092697, 145.1650696, -224.7255096, 219.8510742

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5781835, upper bound: 146.6102108
time: 0.95 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -87.5313339, 119.9613190, -93.8257675, 128.6371765, -216.1685181, 213.7870483
1: -71.6790466, 100.4381485, -76.8269119, 107.9650726, -179.6441040, 177.2650604
2: -58.5908508, 100.8096313, -62.9012985, 108.2000275, -166.7908783, 163.7109375
3: -94.2077713, 121.3282166, -100.9363632, 130.2722931, -224.4800720, 222.2645874
4: -77.7931213, 135.1193237, -83.5092697, 145.1650696, -222.9581909, 218.6285706

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5956644, upper bound: 146.5955993
time: 1.20 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.69 seconds
NS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6645849
NS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6650062
NS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6435124, upper bound: 146.6854790
NS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6435124, upper bound: 146.6867315
NS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6061779, upper bound: 146.5110633
NS_B1_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5555999, upper bound: 146.5095945
NS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6448589, upper bound: 146.6835662
NS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6448589, upper bound: 146.7033496
NS_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5327808, upper bound: 146.6225380
NS_B1_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5309511, upper bound: 146.5562047
NS_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6439509
NS_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6853101
NS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6472987, upper bound: 146.6481984
NS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6472987, upper bound: 146.6481984
NS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6412062, upper bound: 146.6955956
NS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6412062, upper bound: 146.7039633
NS_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
NS_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
NS_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
NS_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6981470, upper bound: 146.7050442
NS_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5277178, upper bound: 146.6130842
NS_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5264679, upper bound: 146.5702614
NS_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6763510, upper bound: 146.6395453
NS_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6982580, upper bound: 146.7054633
NS_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6334123
NS_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6337493
NS_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
NS_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5801684, upper bound: 146.6442964
NS_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5734511, upper bound: 146.5933762
NS_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5734511, upper bound: 146.5933762
NS_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6090476
NS_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5791316, upper bound: 146.6090476
NS_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
NS_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
NS_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
NS_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.7050442, upper bound: 146.6981470
NS_B2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6866469
NS_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6865168
NS_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6425322, upper bound: 146.5731233
NS_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5994463, upper bound: 146.5721441
NS_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6334123, upper bound: 146.5674976
NS_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5730509
NS_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6442964, upper bound: 146.5801684
NS_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6337493, upper bound: 146.5801684
NS_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5734511
NS_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5730509
NS_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6090476, upper bound: 146.5791316
NS_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6090476, upper bound: 146.5791316
NS_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7020957
NS_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6798467, upper bound: 146.7023709
NS_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7020957
NS_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7023709
NS_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7046115
NS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6797615, upper bound: 146.7046115
NS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6798467, upper bound: 146.7047320
NS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6798467, upper bound: 146.7047320
NS_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6564929, upper bound: 146.5973341
NS_B2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6564929, upper bound: 146.5973341
NS_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
NS_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191323
NS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5781835, upper bound: 146.6102108
NS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740
NS_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.5956644, upper bound: 146.5955993
NS_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.69
Output dim: 0, lower bound: -146.6177740, upper bound: 146.6177740

## BFS NS instance: NS_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -45.5058365, 59.1900978, -63.4164619, 83.2889862, -128.7948303, 122.6065598
1: -34.8902168, 48.7785645, -48.9800644, 68.7593842, -103.6495972, 97.7585983
2: -28.8770924, 49.6039162, -40.7057571, 70.1618729, -99.0389481, 90.3096695
3: -45.9418869, 58.9198456, -64.1606293, 82.7382889, -128.6801605, 123.0804749
4: -38.5053291, 65.5943680, -54.1929169, 93.2696152, -131.7749329, 119.7872849

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6645849
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6645849
time: 0.65 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -40.6148300, 52.1892014, -63.4164619, 83.2889862, -123.9038086, 115.6056671
1: -30.9134922, 42.6595116, -48.9800644, 68.7593842, -99.6728745, 91.6395569
2: -25.5832329, 43.3156853, -40.7057571, 70.1618729, -95.7451019, 84.0214310
3: -40.4985733, 51.6264191, -64.1606293, 82.7382889, -123.2368546, 115.7870483
4: -33.9366722, 57.3073654, -54.1929169, 93.2696152, -127.2062836, 111.5002823

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6650062
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6650062
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -45.5058365, 59.1900978, -49.0284653, 63.1552391, -108.6610718, 108.2185669
1: -34.8902168, 48.7785645, -37.4323120, 51.8843918, -86.7746048, 86.2108612
2: -28.8770924, 49.6039162, -31.0997543, 52.6751099, -81.5521851, 80.7036591
3: -45.9418869, 58.9198456, -48.8514748, 62.6909065, -108.6327820, 107.7713165
4: -38.5053291, 65.5943680, -41.3561020, 69.7939453, -108.2992706, 106.9504700

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6854790
time: 0.77 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6854790
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -40.6148300, 52.1892014, -49.0284653, 63.1552391, -103.7700653, 101.2176666
1: -30.9134922, 42.6595116, -37.4323120, 51.8843918, -82.7978821, 80.0918121
2: -25.5832329, 43.3156853, -31.0997543, 52.6751099, -78.2583466, 74.4154358
3: -40.4985733, 51.6264191, -48.8514748, 62.6909065, -103.1894836, 100.4778900
4: -33.9366722, 57.3073654, -41.3561020, 69.7939453, -103.7306061, 98.6634674

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6867315
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6077374, upper bound: 146.6867315
time: 1.05 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -48.5949402, 63.6081314, -51.2175674, 67.1119308, -115.7068710, 114.8256912
1: -37.7306786, 52.4548111, -40.2368126, 55.1920204, -92.9226990, 92.6916122
2: -31.1921425, 53.2458687, -33.2247581, 55.8661804, -87.0583191, 86.4706268
3: -49.6124077, 63.3990364, -52.5563812, 66.6969986, -116.3094025, 115.9554138
4: -41.5441780, 70.5757675, -44.1320305, 74.4198837, -115.9640656, 114.7077866

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6050784, upper bound: 146.5109387
time: 0.90 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6050784, upper bound: 146.5110633
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -43.5288582, 56.4985695, -68.2685165, 90.0284348, -133.5572968, 124.7670822
1: -33.7135353, 46.1598244, -53.5066185, 74.2120667, -107.9255981, 99.6664352
2: -27.8262539, 46.7750587, -44.3684959, 75.5368500, -103.3631058, 91.1435394
3: -44.0973091, 55.9171486, -70.0045929, 89.2658615, -133.3631744, 125.9217377
4: -36.9151611, 62.0547600, -58.9532394, 100.8943100, -137.8094788, 121.0079956

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6381320, upper bound: 146.6777375
time: 0.86 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6448589, upper bound: 146.6835662
time: 1.18 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -43.5288582, 56.4985695, -47.8477554, 62.3074989, -105.8363571, 104.3463211
1: -33.7135353, 46.1598244, -37.1600494, 50.8735123, -84.5870514, 83.3198624
2: -27.8262539, 46.7750587, -30.6940975, 51.8155975, -79.6418457, 77.4691391
3: -44.0973091, 55.9171486, -48.5689278, 61.5174103, -105.6147156, 104.4860764
4: -36.9151611, 62.0547600, -40.7375488, 68.8575211, -105.7726746, 102.7923126

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6076438, upper bound: 146.6881176
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6076438, upper bound: 146.7031909
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -48.2632217, 62.1985207, -68.5043335, 90.0341568, -138.2973480, 130.7028503
1: -37.4914589, 51.1904297, -53.0773087, 74.3221359, -111.8135986, 104.2677307
2: -31.0026722, 51.5841599, -44.1364021, 75.7964706, -106.7991409, 95.7205582
3: -48.7440987, 62.0382957, -69.4603424, 89.3869934, -138.1310730, 131.4986420
4: -41.1413651, 68.4065170, -58.7091751, 100.9141769, -142.0555267, 127.1156921

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5325792, upper bound: 146.6179256
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5325792, upper bound: 146.6225380
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -70.3638153, 92.4291382, -73.9281998, 97.8744202, -168.2381897, 166.3573151
1: -55.1202660, 76.1830978, -58.1919441, 80.5927505, -135.7130127, 134.3750458
2: -45.7413101, 77.6045532, -48.1970291, 82.0537949, -127.7951050, 125.8015823
3: -71.8991776, 91.6738815, -76.1544571, 96.9829102, -168.8820648, 167.8283386
4: -60.6913795, 103.5729828, -64.0241547, 109.7183838, -170.4097595, 167.5971375

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6439509
time: 0.72 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6439509
time: 0.57 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -48.6898956, 62.5317764, -73.9281998, 97.8744202, -146.5643158, 136.4599609
1: -37.5633392, 51.1040039, -58.1919441, 80.5927505, -118.1560898, 109.2959442
2: -31.1202602, 51.7925301, -48.1970291, 82.0537949, -113.1740570, 99.9895630
3: -48.7319031, 61.9018021, -76.1544571, 96.9829102, -145.7147827, 138.0562592
4: -41.1979332, 68.6522675, -64.0241547, 109.7183838, -150.9162903, 132.6764221

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6367477, upper bound: 146.6752664
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6439509, upper bound: 146.6853101
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -70.3638153, 92.4291382, -43.5288582, 56.4985695, -126.8623734, 135.9579926
1: -55.1202660, 76.1830978, -33.7135353, 46.1598244, -101.2800827, 109.8966293
2: -45.7413101, 77.6045532, -27.8262539, 46.7750587, -92.5163727, 105.4308090
3: -71.8991776, 91.6738815, -44.0973091, 55.9171486, -127.8163300, 135.7711945
4: -60.6913795, 103.5729828, -36.9151611, 62.0547600, -122.7461395, 140.4881439

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6650062, upper bound: 146.6077893
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6835662, upper bound: 146.6448589
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -70.3638153, 92.4291382, -48.6898956, 62.5317764, -132.8955841, 141.1190338
1: -55.1202660, 76.1830978, -37.5633392, 51.1040039, -106.2242737, 113.7464371
2: -45.7413101, 77.6045532, -31.1202602, 51.7925301, -97.5338440, 108.7248154
3: -71.8991776, 91.6738815, -48.7319031, 61.9018021, -133.8009796, 140.4057617
4: -60.6913795, 103.5729828, -41.1979332, 68.6522675, -129.3436432, 144.7708893

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6794836, upper bound: 146.6412062
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6854848, upper bound: 146.6481984
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -58.0203781, 75.2546082, -52.4048882, 68.4087601, -126.4291306, 127.6595001
1: -45.8815002, 62.2253799, -41.0123634, 55.9361687, -101.8176651, 103.2377396
2: -37.9879379, 61.8730507, -33.8807259, 56.8105774, -94.7985153, 95.7537766
3: -59.2922478, 75.5359955, -53.4349403, 67.6620789, -126.9543304, 128.9709320
4: -50.1904144, 82.5301971, -44.8779449, 75.6150970, -125.8055115, 127.4081421

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6582064, upper bound: 146.6430157
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6969462, upper bound: 146.6923719
time: 1.24 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -46.7513084, 59.8868179, -55.3256187, 72.3050690, -119.0563812, 115.2124252
1: -36.0112381, 48.9019012, -43.4013748, 59.2114868, -95.2227249, 92.3032761
2: -29.8367023, 49.5287552, -35.8692703, 59.9859810, -89.8226776, 85.3980179
3: -46.7223549, 59.2589226, -56.5503922, 71.6232834, -118.3456345, 115.8093109
4: -39.4825897, 65.6364441, -47.5470848, 79.9374924, -119.4200821, 113.1835251

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038979, upper bound: 146.7039480
time: 1.02 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038979, upper bound: 146.7039627
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -78.8680801, 106.8411484, -58.1859283, 76.2957153, -155.1637878, 165.0270691
1: -64.2956390, 89.9398575, -46.3820801, 62.9604378, -127.2560654, 136.3219299
2: -52.8499603, 89.4106674, -38.2315903, 62.8689003, -115.7188492, 127.6422577
3: -84.4407883, 108.6659012, -60.1663933, 76.4167404, -160.8574829, 168.8322906
4: -69.9198761, 120.0836792, -50.5320625, 83.7994766, -153.7193604, 170.6157227

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053259, upper bound: 146.7042664
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053259, upper bound: 146.7054097
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -79.0112152, 107.2077866, -58.1859283, 76.2957153, -155.3069305, 165.3937073
1: -64.3562012, 90.3239746, -46.3820801, 62.9604378, -127.3166351, 136.7060547
2: -52.8432465, 89.6180801, -38.2315903, 62.8689003, -115.7121353, 127.8496704
3: -84.6326141, 109.1384201, -60.1663933, 76.4167404, -161.0493469, 169.3047791
4: -69.9387360, 120.3619919, -50.5320625, 83.7994766, -153.7382202, 170.8940582

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053259, upper bound: 146.7042664
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7053259, upper bound: 146.7054097
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -78.8680801, 106.8411484, -63.2701683, 82.4207993, -161.2888794, 170.1113129
1: -64.2956390, 89.9398575, -50.2276192, 68.2725677, -132.5682068, 140.1674500
2: -52.8499603, 89.4106674, -41.5368042, 67.8949814, -120.7449265, 130.9474792
3: -84.4407883, 108.6659012, -64.9713974, 82.8786316, -167.3194275, 173.6372986
4: -69.9198761, 120.0836792, -54.8750114, 90.6103745, -160.5302429, 174.9586487

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6924276, upper bound: 146.7038153
time: 0.69 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6981431, upper bound: 146.7050418
time: 1.09 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -79.0112152, 107.2077866, -63.2701683, 82.4207993, -161.4320068, 170.4779510
1: -64.3562012, 90.3239746, -50.2276192, 68.2725677, -132.6287689, 140.5515747
2: -52.8432465, 89.6180801, -41.5368042, 67.8949814, -120.7382126, 131.1548767
3: -84.6326141, 109.1384201, -64.9713974, 82.8786316, -167.5112457, 174.1098022
4: -69.9387360, 120.3619919, -54.8750114, 90.6103745, -160.5490875, 175.2369995

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6924276, upper bound: 146.7038154
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6924276, upper bound: 146.7050418
time: 1.12 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -84.1679382, 112.6055832, -58.1859283, 76.2957153, -160.4636536, 170.7914886
1: -68.1102982, 94.7352829, -46.3820801, 62.9604378, -131.0707397, 141.1173706
2: -55.9914970, 93.9249191, -38.2315903, 62.8689003, -118.8603897, 132.1565094
3: -88.9506454, 114.5857086, -60.1663933, 76.4167404, -165.3673859, 174.7520752
4: -73.9732666, 125.8390121, -50.5320625, 83.7994766, -157.7727356, 176.3710785

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5236839, upper bound: 146.5816014
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5277178, upper bound: 146.6130842
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -79.5768585, 106.1595764, -60.4968147, 78.5101471, -158.0870056, 166.6563873
1: -63.9160995, 89.2002640, -47.7909660, 64.9776535, -128.8937378, 136.9912262
2: -52.6802750, 88.4765930, -39.5829697, 64.5988388, -117.2790985, 128.0595398
3: -83.5357056, 107.8305359, -61.7850723, 78.8629074, -162.3985901, 169.6156006
4: -69.6460648, 118.4835205, -52.2948570, 86.1725159, -155.8185730, 170.7783813

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5659979, upper bound: 146.6259646
time: 0.73 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5664940, upper bound: 146.5818354
time: 0.60 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -84.4672012, 113.0568924, -63.1267204, 82.2301559, -166.6973572, 176.1835785
1: -68.3404999, 95.0867844, -50.1075439, 68.1107941, -136.4512787, 145.1943359
2: -56.1803055, 94.3147812, -41.4384766, 67.7360840, -123.9163818, 135.7532654
3: -89.2645187, 115.0154343, -64.8175430, 82.6809692, -171.9454956, 179.8329773
4: -74.2281265, 126.3548813, -54.7460327, 90.3969345, -164.6250458, 181.1008759

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5731298, upper bound: 146.6425484
time: 0.79 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5721450, upper bound: 146.5995050
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -90.4219818, 121.9592972, -56.4019470, 73.7915802, -164.2135620, 178.3612366
1: -73.1199570, 102.5635071, -44.2301216, 60.7880516, -133.9079895, 146.7936249
2: -60.1868172, 102.0159378, -36.6528969, 61.1218834, -121.3086853, 138.6688385
3: -95.8767319, 123.8952942, -57.7706604, 73.4784851, -169.3552246, 181.6659241
4: -79.7158279, 136.8176270, -48.6710129, 81.6132889, -161.3291168, 185.4886475

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6334123
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6334123
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -90.4219818, 121.9592972, -60.2948952, 79.1271286, -169.5491028, 182.2541962
1: -73.1199570, 102.5635071, -47.7550697, 65.2957764, -138.4157104, 150.3185730
2: -60.1868172, 102.0159378, -39.4233818, 65.6054153, -125.7922211, 141.4393158
3: -95.8767319, 123.8952942, -62.2565231, 79.0400467, -174.9167786, 186.1518097
4: -79.7158279, 136.8176270, -52.2891884, 87.6184921, -167.3343048, 189.1068115

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6337493
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5674976, upper bound: 146.6337493
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -94.8670349, 128.3668213, -59.3583717, 77.8945007, -172.7615356, 187.7251892
1: -77.2229843, 107.9733505, -47.0230522, 64.2843628, -141.5073395, 154.9963989
2: -63.3891525, 107.4281540, -38.7983780, 64.5626373, -127.9517899, 146.2265320
3: -101.1763992, 130.4962311, -61.2819099, 77.8366852, -179.0130920, 191.7781219
4: -83.9320221, 144.1323700, -51.4784241, 86.2115021, -170.1435089, 195.6107788

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5675703, upper bound: 146.6363088
time: 0.64 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5675703, upper bound: 146.6442964
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -94.8670349, 128.3668213, -58.8321114, 78.3008499, -173.1678619, 187.1989288
1: -77.2229843, 107.9733505, -46.8591003, 64.5102463, -141.7332306, 154.8324280
2: -63.3891525, 107.4281540, -38.6052208, 65.3042603, -128.6934204, 146.0333405
3: -101.1763992, 130.4962311, -61.2504730, 77.8967743, -179.0731354, 191.7467041
4: -83.9320221, 144.1323700, -51.2711067, 87.3223801, -171.2543640, 195.4034576

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5675703, upper bound: 146.6363087
time: 1.03 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5675703, upper bound: 146.6442964
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -86.9514771, 119.0431976, -56.4793358, 73.7433548, -160.6948242, 175.5225067
1: -70.7517090, 99.9098358, -44.4461594, 60.7963104, -131.5480194, 144.3559875
2: -58.0821304, 100.1023331, -36.7450027, 61.0876846, -119.1698151, 136.8473358
3: -93.1455078, 120.4289322, -57.9216232, 73.5881653, -166.7336731, 178.3505402
4: -77.1675339, 134.3264465, -48.7570572, 81.5142822, -158.6817932, 183.0834961

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_B1_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664178, upper bound: 146.5933762
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_B1_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664178, upper bound: 146.5933762
time: 0.58 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -86.9514771, 119.0431976, -56.3274574, 74.6053162, -161.5567474, 175.3706512
1: -70.7517090, 99.9098358, -44.5752983, 61.4515991, -132.2033081, 144.4851379
2: -58.0821304, 100.1023331, -36.7933197, 62.2006302, -120.2827606, 136.8956604
3: -93.1455078, 120.4289322, -58.2525978, 74.1658401, -167.3113403, 178.6814880
4: -77.1675339, 134.3264465, -48.8671799, 83.1371078, -160.3046417, 183.1936188

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_B1_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664178, upper bound: 146.5933762
time: 0.60 seconds

## Relational analysis of NS_B1_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_B1_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664178, upper bound: 146.5933762
time: 1.19 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -92.1462021, 126.4399643, -59.3583717, 77.8945007, -170.0407104, 185.7983398
1: -75.4836807, 106.1654816, -47.0230522, 64.2843628, -139.7680359, 153.1885376
2: -61.8041611, 106.3367386, -38.7983780, 64.5626373, -126.3667984, 145.1351166
3: -99.2066345, 128.0771484, -61.2819099, 77.8366852, -177.0433197, 189.3590546
4: -82.0615311, 142.7024078, -51.4784241, 86.2115021, -168.2730408, 194.1808014

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_B1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664935, upper bound: 146.5984659
time: 0.63 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_B1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664935, upper bound: 146.6090476
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -92.1462021, 126.4399643, -58.8189583, 78.2843018, -170.4305115, 185.2589264
1: -75.4836807, 106.1654816, -46.8484459, 64.4962845, -139.9799652, 153.0139313
2: -61.8041611, 106.3367386, -38.5962296, 65.2905350, -127.0946884, 144.9329681
3: -99.2066345, 128.0771484, -61.2367744, 77.8799896, -177.0866241, 189.3139191
4: -82.0615311, 142.7024078, -51.2593880, 87.3034286, -169.3649597, 193.9617920

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664935, upper bound: 146.5984659
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5664935, upper bound: 146.6090476
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -58.1859283, 76.2957153, -79.2176056, 107.2103958, -165.3963318, 155.5133209
1: -46.3820801, 62.9604378, -64.5331345, 90.2396240, -136.6217041, 127.4935532
2: -38.2315903, 62.8689003, -53.0479469, 89.7292328, -127.9608231, 115.9168320
3: -60.1663933, 76.4167404, -84.7372742, 109.0198288, -169.1861877, 161.1539917
4: -50.5320625, 83.7994766, -70.1789627, 120.4900742, -171.0221405, 153.9784393

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7042664, upper bound: 146.7053259
time: 0.66 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_B1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054097, upper bound: 146.7057928
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -58.1859283, 76.2957153, -79.3594437, 107.5857620, -165.7716675, 155.6551514
1: -46.3820801, 62.9604378, -64.5906601, 90.6275787, -137.0096588, 127.5510864
2: -38.2315903, 62.8689003, -53.0364532, 89.9425659, -128.1741638, 115.9053497
3: -60.1663933, 76.4167404, -84.9271698, 109.4980469, -169.6644440, 161.3438873
4: -50.5320625, 83.7994766, -70.1930618, 120.7732239, -171.3052673, 153.9925385

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7042664, upper bound: 146.7053708
time: 0.62 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054097, upper bound: 146.7057946
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -63.2701683, 82.4207993, -79.2176056, 107.2103958, -170.4805603, 161.6383972
1: -50.2276192, 68.2725677, -64.5331345, 90.2396240, -140.4672394, 132.8056946
2: -41.5368042, 67.8949814, -53.0479469, 89.7292328, -131.2660370, 120.9429169
3: -64.9713974, 82.8786316, -84.7372742, 109.0198288, -173.9912109, 167.6159058
4: -54.8750114, 90.6103745, -70.1789627, 120.4900742, -175.3650818, 160.7893219

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038154, upper bound: 146.6924276
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050418, upper bound: 146.6981431
time: 0.91 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -63.2701683, 82.4207993, -79.3594437, 107.5857620, -170.8559265, 161.7802429
1: -50.2276192, 68.2725677, -64.5906601, 90.6275787, -140.8551941, 132.8632202
2: -41.5368042, 67.8949814, -53.0364532, 89.9425659, -131.4793701, 120.9314194
3: -64.9713974, 82.8786316, -84.9271698, 109.4980469, -174.4694519, 167.8058014
4: -54.8750114, 90.6103745, -70.1930618, 120.7732239, -175.6482239, 160.8034363

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7038154, upper bound: 146.6932766
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050418, upper bound: 146.6981431
time: 1.04 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -55.4315834, 72.3115387, -74.7648926, 99.6353912, -155.0669708, 147.0764160
1: -44.0320549, 59.6672707, -60.2679405, 83.9649277, -127.9969559, 119.9352112
2: -36.3209381, 59.5300941, -49.6179390, 83.0990295, -119.4199677, 109.1480179
3: -57.0480118, 72.4258804, -78.7299118, 101.4767990, -158.5248108, 151.1557465
4: -47.9916916, 79.2743073, -65.5002899, 111.1828690, -159.1745605, 144.7745514

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6865938
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6973670, upper bound: 146.6866469
time: 0.60 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -59.4349022, 77.1058578, -74.7648926, 99.6353912, -159.0702972, 151.8707275
1: -47.0401154, 63.8462601, -60.2679405, 83.9649277, -131.0050354, 124.1141968
2: -38.9214668, 63.4651871, -49.6179390, 83.0990295, -122.0204926, 113.0831223
3: -60.7950249, 77.5004883, -78.7299118, 101.4767990, -162.2718201, 156.2303467
4: -51.4103203, 84.6192780, -65.5002899, 111.1828690, -162.5931854, 150.1195679

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6865168
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7031044, upper bound: 146.6865168
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -70.6798096, 93.0840759, -77.2184067, 103.0455017, -173.7253113, 170.3024902
1: -56.5366020, 77.1314697, -62.4382324, 86.3934708, -142.9300690, 139.5696869
2: -46.6428108, 77.0236740, -51.2256775, 85.8574142, -132.5001984, 128.2493134
3: -73.3837051, 93.5942612, -81.4987869, 104.6427383, -178.0264435, 175.0930481
4: -61.6865921, 102.9384613, -67.6258698, 114.7995911, -176.4861450, 170.5643311

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B2_B2_B1_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6229299, upper bound: 146.5659824
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B1_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6425322, upper bound: 146.5731233
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -68.4224396, 90.2934647, -75.5738373, 102.7434616, -171.1658783, 165.8673096
1: -54.7306137, 74.7609024, -61.6387062, 85.9881744, -140.7187805, 136.3995819
2: -45.1359406, 74.7552643, -50.3980751, 86.1050491, -131.2409515, 125.1533356
3: -71.1074829, 90.6875534, -80.8080826, 103.9631195, -175.0706024, 171.4956207
4: -59.7334557, 99.8899231, -66.8294220, 115.2570343, -174.9904938, 166.7193451

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5817742, upper bound: 146.5664940
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5994463, upper bound: 146.5721441
time: 1.09 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -56.4019470, 73.7915802, -90.9309692, 122.5153961, -178.9172974, 164.7225342
1: -44.2301216, 60.7880516, -73.4620056, 103.0150375, -147.2451172, 134.2500458
2: -36.6528969, 61.1218834, -60.4697571, 102.5000763, -139.1529694, 121.5916443
3: -57.7706604, 73.4784851, -96.3089676, 124.4303055, -182.2009430, 169.7874451
4: -48.6710129, 81.6132889, -80.0910110, 137.4320221, -186.1030273, 161.7042999

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6334123, upper bound: 146.5674976
time: 0.99 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6334123, upper bound: 146.5674976
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -60.2948952, 79.1271286, -90.9309692, 122.5153961, -182.8102875, 170.0580597
1: -47.7550697, 65.2957764, -73.4620056, 103.0150375, -150.7700958, 138.7577515
2: -39.4233818, 65.6054153, -60.4697571, 102.5000763, -141.9234619, 126.0751724
3: -62.2565231, 79.0400467, -96.3089676, 124.4303055, -186.6868286, 175.3490143
4: -52.2891884, 87.6184921, -80.0910110, 137.4320221, -189.7212067, 167.7095032

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6334123, upper bound: 146.5730509
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5730509
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -59.3583717, 77.8945007, -95.3544235, 128.9032898, -188.2616577, 173.2489319
1: -47.0230522, 64.2843628, -77.5516357, 108.4070740, -155.4301300, 141.8359985
2: -38.7983780, 64.5626373, -63.6608925, 107.8943710, -146.6927185, 128.2235260
3: -61.2819099, 77.8366852, -101.5921097, 131.0086975, -192.2905884, 179.4287872
4: -51.4784241, 86.2115021, -84.2938538, 144.7262115, -196.2046051, 170.5053558

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5674922
time: 1.22 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5800310
time: 1.14 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -58.8321114, 78.3008499, -95.3544235, 128.9032898, -187.7353821, 173.6552277
1: -46.8591003, 64.5102463, -77.5516357, 108.4070740, -155.2661591, 142.0618896
2: -38.6052208, 65.3042603, -63.6608925, 107.8943710, -146.4995270, 128.9651489
3: -61.2504730, 77.8967743, -101.5921097, 131.0086975, -192.2591705, 179.4888153
4: -51.2711067, 87.3223801, -84.2938538, 144.7262115, -195.9972839, 171.6162415

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5675703
time: 0.68 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6327058, upper bound: 146.5801582
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -56.4793358, 73.7433548, -87.2007294, 119.3170547, -175.7963867, 160.9440765
1: -44.4461594, 60.7963104, -70.9196091, 100.1323242, -144.5784912, 131.7158966
2: -36.7450027, 61.0876846, -58.2212296, 100.3403320, -137.0853271, 119.3089066
3: -57.9216232, 73.5881653, -93.3581390, 120.6934967, -178.6150970, 166.9463043
4: -48.7570572, 81.5142822, -77.3527527, 134.6280518, -183.3851013, 158.8670349

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5655841, upper bound: 146.5664178
time: 1.09 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_A1_A2

### Relational analysis result of NS_B2_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5734511
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -56.3274574, 74.6053162, -87.2007294, 119.3170547, -175.6445160, 161.8059998
1: -44.5752983, 61.4515991, -70.9196091, 100.1323242, -144.7076263, 132.3711700
2: -36.7933197, 62.2006302, -58.2212296, 100.3403320, -137.1336517, 120.4218521
3: -58.2525978, 74.1658401, -93.3581390, 120.6934967, -178.9460449, 167.5239868
4: -48.8671799, 83.1371078, -77.3527527, 134.6280518, -183.4952393, 160.4898682

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5655841, upper bound: 146.5664178
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5730509
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -59.3583717, 77.8945007, -92.3872986, 126.7017288, -186.0601044, 170.2817993
1: -47.0230522, 64.2843628, -75.6435089, 106.3778534, -153.4009094, 139.9278717
2: -38.7983780, 64.5626373, -61.9371109, 106.5636292, -145.3619843, 126.4997482
3: -61.2819099, 77.8366852, -99.4087219, 128.3294067, -189.6113129, 177.2454071
4: -51.4784241, 86.2115021, -82.2395782, 142.9899902, -194.4684143, 168.4510803

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5655841, upper bound: 146.5664935
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -146.5655841, upper bound: 146.5791316
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -58.8189583, 78.2843018, -92.3872986, 126.7017288, -185.5206909, 170.6715546
1: -46.8484459, 64.4962845, -75.6435089, 106.3778534, -153.2263031, 140.1397858
2: -38.5962296, 65.2905350, -61.9371109, 106.5636292, -145.1598511, 127.2276154
3: -61.2367744, 77.8799896, -99.4087219, 128.3294067, -189.5661774, 177.2887115
4: -51.2593880, 87.3034286, -82.2395782, 142.9899902, -194.2493744, 169.5429840

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5664935
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.5933762, upper bound: 146.5791316
time: 0.65 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -72.3207779, 97.6563263, -72.3207779, 97.6563263, -169.9770660, 169.9770660
1: -58.4869423, 82.3626709, -58.4869423, 82.3626709, -140.8496094, 140.8496094
2: -48.1558838, 81.7584534, -48.1558838, 81.7584534, -129.9143066, 129.9143066
3: -76.9195251, 99.3732681, -76.9195251, 99.3732681, -176.2927856, 176.2927856
4: -63.7550850, 109.6183014, -63.7550850, 109.6183014, -173.3733826, 173.3733826

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7043236, upper bound: 146.7046617
time: 0.98 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054609, upper bound: 146.7053474
time: 1.20 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -76.6127090, 102.1225204, -72.3207779, 97.6563263, -174.2690430, 174.4432983
1: -61.7318764, 85.9944458, -58.4869423, 82.3626709, -144.0945435, 144.4813843
2: -50.8281174, 85.2267532, -48.1558838, 81.7584534, -132.5865326, 133.3826294
3: -80.6219482, 103.9286118, -76.9195251, 99.3732681, -179.9952087, 180.8481445
4: -67.1066132, 114.0265732, -63.7550850, 109.6183014, -176.7249146, 177.7816315

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7048595, upper bound: 146.7039899
time: 1.00 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054609, upper bound: 146.7058297
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -72.3207779, 97.6563263, -75.0877838, 101.5556335, -173.8764038, 172.7440491
1: -58.4869423, 82.3626709, -61.0108986, 85.2025681, -143.6894836, 143.3735657
2: -48.1558838, 81.7584534, -50.0403366, 84.9559402, -133.1118164, 131.7987518
3: -76.9195251, 99.3732681, -80.2089539, 103.0284119, -179.9479218, 179.5822144
4: -63.7550850, 109.6183014, -66.1934967, 113.9446564, -177.6997375, 175.8117981

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_B2_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -76.6127090, 102.1225204, -75.0877838, 101.5556335, -178.1683350, 177.2102814
1: -61.7318764, 85.9944458, -61.0108986, 85.2025681, -146.9344177, 147.0053406
2: -50.8281174, 85.2267532, -50.0403366, 84.9559402, -135.7840576, 135.2670746
3: -80.6219482, 103.9286118, -80.2089539, 103.0284119, -183.6503448, 184.1375732
4: -67.1066132, 114.0265732, -66.1934967, 113.9446564, -181.0512695, 180.2200623

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

## BFS NS instance: NS_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -72.3207779, 97.6563263, -76.6164932, 102.1269913, -174.4477539, 174.2727661
1: -58.4869423, 82.3626709, -61.7346001, 85.9981766, -144.4850922, 144.0972748
2: -48.1558838, 81.7584534, -50.8304214, 85.2304993, -133.3863831, 132.5888519
3: -76.9195251, 99.3732681, -80.6253738, 103.9331131, -180.8526306, 179.9986267
4: -63.7550850, 109.6183014, -67.1096649, 114.0315094, -177.7865906, 176.7279663

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6924468, upper bound: 146.7042592
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6957464, upper bound: 146.7046621
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -72.3207779, 97.6563263, -80.7465363, 107.8700638, -180.1908417, 178.4028168
1: -58.4869423, 82.3626709, -65.2797165, 90.3707581, -148.8576813, 147.6423950
2: -48.1558838, 81.7584534, -53.5639725, 89.9739838, -138.1298676, 135.3223877
3: -76.9195251, 99.3732681, -85.2038345, 109.4544983, -186.3740234, 184.5771027
4: -63.7550850, 109.6183014, -70.7425308, 120.2771454, -184.0322266, 180.3608398

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6924468, upper bound: 146.7042592
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6924468, upper bound: 146.7046621
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -76.6164932, 102.1269913, -76.6164932, 102.1269913, -178.7434692, 178.7434692
1: -61.7346001, 85.9981766, -61.7346001, 85.9981766, -147.7327728, 147.7327728
2: -50.8304214, 85.2304993, -50.8304214, 85.2304993, -136.0609131, 136.0609131
3: -80.6253738, 103.9331131, -80.6253738, 103.9331131, -184.5584869, 184.5584869
4: -67.1096649, 114.0315094, -67.1096649, 114.0315094, -181.1411591, 181.1411591

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6904499, upper bound: 146.7037702
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6962331, upper bound: 146.7047320
time: 1.14 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -76.6164932, 102.1269913, -80.7465363, 107.8700638, -184.4865570, 182.8735046
1: -61.7346001, 85.9981766, -65.2797165, 90.3707581, -152.1053619, 151.2778931
2: -50.8304214, 85.2304993, -53.5639725, 89.9739838, -140.8044128, 138.7944641
3: -80.6253738, 103.9331131, -85.2038345, 109.4544983, -190.0798645, 189.1369476
4: -67.1096649, 114.0315094, -70.7425308, 120.2771454, -187.3867950, 184.7740326

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6858997, upper bound: 146.7032895
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6962331, upper bound: 146.7047320
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -85.7774658, 115.4548492, -93.7582779, 126.6526413, -212.4301147, 209.2131042
1: -69.7238922, 96.7863235, -76.2749176, 106.5015335, -176.2254333, 173.0612335
2: -57.2412758, 96.3669128, -62.5941658, 105.9324646, -163.1737366, 158.9610748
3: -91.2028275, 117.1307983, -99.9107895, 128.7380829, -219.9409027, 217.0415955
4: -75.7288361, 129.2347412, -82.8577957, 142.0824280, -217.8112640, 212.0925140

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6388433, upper bound: 146.5897176
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6564929, upper bound: 146.5973341
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -84.4778137, 115.5848083, -93.7582779, 126.6526413, -211.1304474, 209.3430786
1: -69.3032761, 96.6761322, -76.2749176, 106.5015335, -175.8048096, 172.9510193
2: -56.7338409, 97.0464859, -62.5941658, 105.9324646, -162.6663055, 159.6406403
3: -90.9858398, 116.8325043, -99.9107895, 128.7380829, -219.7239227, 216.7432861
4: -75.1933136, 130.2967224, -82.8577957, 142.0824280, -217.2757416, 213.1544952

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6388433, upper bound: 146.5897176
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6564929, upper bound: 146.5973341
time: 1.36 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -86.2700882, 116.2182999, -95.7246552, 129.3341522, -215.6042175, 211.9429474
1: -70.1226425, 97.5188293, -77.8616943, 108.7375565, -178.8601532, 175.3804626
2: -57.4733582, 97.0775452, -63.9038506, 108.2392349, -165.7125854, 160.9813690
3: -91.8090591, 118.0362396, -101.9636459, 131.4375763, -223.2466278, 219.9998779
4: -76.0135422, 130.0392914, -84.5994949, 145.1608429, -221.1743774, 214.6387939

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6611870, upper bound: 146.6130245
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191315
time: 0.79 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -83.9049835, 114.9982071, -95.7246552, 129.3341522, -213.2391052, 210.7228394
1: -68.7663040, 96.3420868, -77.8616943, 108.7375565, -177.5038452, 174.2037659
2: -56.1686440, 96.5717010, -63.9038506, 108.2392349, -164.4078827, 160.4755554
3: -90.4081726, 116.4327698, -101.9636459, 131.4375763, -221.8457184, 218.3964233
4: -74.5785599, 129.4405823, -84.5994949, 145.1608429, -219.7394104, 214.0400696

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6611870, upper bound: 146.6130245
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6786219, upper bound: 146.6191315
time: 0.69 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.59 + 416.99 = 420.58 seconds
